"""Create additive judge-facing artifacts from saved pipeline results."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from scipy.optimize import minimize

from src.customer_pipeline.stage2 import _qaoa_circuit, _vectorized_qaoa_objective
from src.customer_pipeline.stage3 import exact_at_least_k


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _transpile_stats(circuit, backend, optimization_level, seed_transpiler):
    compiled = transpile(circuit, backend, optimization_level=optimization_level,
                         seed_transpiler=seed_transpiler,
                         initial_layout=list(range(circuit.num_qubits)))
    ops = {str(k): int(v) for k, v in compiled.count_ops().items()}
    logical_to_physical = {str(i): i for i in range(circuit.num_qubits)}
    if compiled.layout is not None:
        try:
            logical_to_physical = {str(i): int(compiled.layout.final_index_layout()[i])
                                   for i in range(circuit.num_qubits)}
        except Exception:
            pass
    twoq_names = {"cx", "cz", "ecr", "rxx", "ryy", "iswap", "swap"}
    return compiled, {
        "backend": "Qiskit AerSimulator with a linear coupling map (simulated topology, not an IBM device)",
        "optimization_level": int(optimization_level), "seed_transpiler": int(seed_transpiler),
        "logical_qubits": int(circuit.num_qubits), "physical_qubits": int(len(set(logical_to_physical.values()))),
        "compiled_circuit_register_width": int(compiled.num_qubits),
        "logical_to_physical_qubit_mapping": logical_to_physical,
        "transpiled_depth": int(compiled.depth()), "gate_counts_by_type": ops,
        "two_qubit_gate_count": int(sum(v for name, v in ops.items() if name in twoq_names)),
        "basis_gates": ["rz", "sx", "x", "cx"],
    }


def _stage3_circuit(probabilities):
    n = len(probabilities)
    circuit = QuantumCircuit(n, n)
    for q, probability in enumerate(probabilities):
        circuit.ry(2 * np.arcsin(np.sqrt(float(probability))), q)
        circuit.measure(q, q)
    return circuit


def _wilson(successes, shots, z=1.959963984540054):
    phat = successes / shots
    den = 1 + z*z/shots
    center = (phat + z*z/(2*shots))/den
    half = z*np.sqrt(phat*(1-phat)/shots + z*z/(4*shots*shots))/den
    return [float(center-half), float(center+half)]


def _noise_run(probabilities, seed, shots, optimization_level):
    n = len(probabilities)
    coupling = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["sx", "x"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    assignment = np.asarray([[0.98, 0.03], [0.02, 0.97]], dtype=float)
    noise.add_all_qubit_readout_error(ReadoutError(assignment.T.tolist()))
    backend = AerSimulator(method="matrix_product_state", coupling_map=coupling,
                           basis_gates=["rz", "sx", "x", "cx"], noise_model=noise)
    compiled, stats = _transpile_stats(_stage3_circuit(probabilities), backend,
                                       optimization_level, seed)
    counts = backend.run(compiled, shots=int(shots), seed_simulator=int(seed)).result().get_counts()
    success = sum(c for bits, c in counts.items() if int(bits.replace(" ", ""), 2).bit_count() >= 3)
    unmitigated = success / shots

    # Tensor-product assignment inversion using the stated readout channel.
    full_assignment = assignment
    for _ in range(n-1):
        full_assignment = np.kron(assignment, full_assignment)
    observed = np.zeros(1 << n, dtype=float)
    for bits, count in counts.items():
        observed[int(bits.replace(" ", "")[::-1], 2)] += count / shots
    corrected = np.linalg.solve(full_assignment, observed)
    success_mask = np.asarray([index.bit_count() >= 3 for index in range(1 << n)], dtype=float)
    mitigated = float(success_mask @ corrected)
    weights = np.linalg.solve(full_assignment.T, success_mask)
    mean_weight = float(weights @ observed)
    second = float((weights * weights) @ observed)
    mitigated_se = float(np.sqrt(max(0.0, second - mean_weight*mean_weight) / shots))
    mitigated_ci = [max(0.0, mitigated - 1.96*mitigated_se), min(1.0, mitigated + 1.96*mitigated_se)]
    return {
        "status": "completed", "noise_model": {"single_qubit_depolarizing_probability": 0.005,
          "cx_depolarizing_probability": 0.02, "readout_assignment_matrix_true_columns_observed_rows": assignment.tolist()},
        "transpilation": stats, "shots": int(shots), "seed_simulator": int(seed),
        "measurement_counts": {str(k): int(v) for k, v in counts.items()},
        "exact_probability": exact_at_least_k(probabilities, 3),
        "unmitigated_probability": float(unmitigated), "unmitigated_wilson_95_ci": _wilson(success, shots),
        "unmitigated_absolute_error": float(abs(unmitigated-exact_at_least_k(probabilities, 3))),
        "unmitigated_relative_error": float(abs(unmitigated-exact_at_least_k(probabilities, 3))/exact_at_least_k(probabilities, 3)),
        "readout_mitigated_probability_linear_inversion": mitigated,
        "mitigated_approximate_95_ci_sampling_only": mitigated_ci,
        "mitigated_absolute_error": float(abs(mitigated-exact_at_least_k(probabilities, 3))),
        "mitigated_relative_error": float(abs(mitigated-exact_at_least_k(probabilities, 3))/exact_at_least_k(probabilities, 3)),
        "mitigation_note": "Tensor-product readout assignment inversion; CI accounts for finite measurement sampling only, not calibration uncertainty.",
    }


def _stage2_transpilation(stage2, optimization_level, seed_transpiler):
    candidates = pd.DataFrame(stage2["candidate_rows"])
    net = candidates["expected_net_value"].to_numpy(float)
    record = stage2["qaoa"]["best_optimization_run"]
    if int(record["seed"]) != int(stage2["qaoa"]["best_of_80_measured_runs"]["best_run_settings"]["seed"]):
        raise ValueError("Saved best-of-80 settings do not match the persisted optimizer parameters.")
    p = int(record["p"])
    theta = np.asarray(record["parameters"], dtype=float)
    circuit = _qaoa_circuit(net, p, theta[:p], theta[p:], measure=True)
    coupling = [[i, i+1] for i in range(len(net)-1)] + [[i+1, i] for i in range(len(net)-1)]
    backend = AerSimulator(coupling_map=coupling, basis_gates=["rz", "sx", "x", "cx"])
    compiled, stats = _transpile_stats(circuit, backend, optimization_level, seed_transpiler)
    stats.update({"best_of_80_sampled_mass_run": stage2["qaoa"]["best_of_80_measured_runs"]["best_run_settings"],
                  "saved_optimizer_parameters_used": True,
                  "best_expected_normalized_objective": float(record["best_expected_normalized_objective"]),
                  "measurement_shots_for_original_reported_run": int(stage2["qaoa"]["shots"]),
                  "note": "Transpiled the exact persisted parameters from the best-of-80 measured run; this is a diagnostic, not a new QAOA score or hardware run."})
    return stats

def _uncertainty_sensitivity(top, calibration_csv, selected, exact_probability, budget_limit, capacity_limit, seed=42, draws=10000):
    calibration = pd.read_csv(calibration_csv)
    row = calibration.loc[calibration["bin"].astype(str).eq("top_10_percent")].iloc[0]
    mean_p = float(row["mean_predicted"])
    delta_low = float(row["ci95_low"] - mean_p)
    delta_high = float(row["ci95_high"] - mean_p)
    ids = top["customer_id"].astype(str).to_numpy()
    p = top["predicted_probability"].to_numpy(float)
    deposit = top["expected_deposit_value"].to_numpy(float)
    costs = top["contact_cost"].to_numpy(float)
    capacities = top["capacity_units"].to_numpy(int)
    subsets = np.asarray(list(combinations(range(len(top)), 6)), dtype=int)
    feasible = np.asarray([(costs[s].sum() <= int(budget_limit) and capacities[s].sum() <= int(capacity_limit)) for s in subsets])
    subsets = subsets[feasible]
    rng = np.random.default_rng(int(seed))
    offsets = rng.uniform(delta_low, delta_high, size=int(draws))
    objective_values = np.clip(p[None, :] + offsets[:, None], 0, 1) * deposit[None, :] - costs[None, :]
    subset_objectives = objective_values @ np.eye(len(ids))[subsets].sum(axis=1).T
    winners = np.argmax(subset_objectives, axis=1)
    selected_ids = [str(x) for x in selected["customer_id"]]
    chosen_same = np.asarray([set(ids[subsets[i]]) == set(selected_ids) for i in winners])
    selected_indices = [np.where(ids == cid)[0][0] for cid in selected_ids]
    stage3_vals = np.asarray([
        exact_at_least_k(np.clip(p[selected_indices] + offset, 0, 1), 3)
        for offset in offsets
    ])
    return {
        "method": "shared additive calibration-bin uncertainty perturbation; top_10_percent OOF calibration CI is applied as a common offset to all candidate probabilities",
        "warning": "This is a group-level stress-test proxy, not a customer-specific test probability confidence interval; test labels are unavailable.",
        "seed": int(seed), "draws": int(draws), "calibration_bin": "top_10_percent",
        "calibration_mean_prediction": mean_p, "calibration_observed_rate_ci95": [float(row["ci95_low"]), float(row["ci95_high"])],
        "sampled_common_probability_offset_range": [delta_low, delta_high],
        "stage2_selection_matches_submitted_exact_selection_rate": float(chosen_same.mean()),
        "most_frequent_selected_set": [str(ids[i]) for i in subsets[int(np.bincount(winners, minlength=len(subsets)).argmax())]],
        "stage3_exact_probability_from_perturbed_probabilities_quantiles_2_5_50_97_5": [float(x) for x in np.quantile(stage3_vals, [.025,.5,.975])],
        "original_stage3_exact_probability": float(exact_probability),
    }


def export_submission_artifacts(package_root=None, repository_root=None, noise_shots=20000, mc_draws=1000000,
                                uncertainty_draws=10000, optimization_level=1, seed_transpiler=42):
    package = Path(package_root or Path(__file__).resolve().parents[2]).resolve()
    repo = Path(repository_root or package).resolve()
    result_dir = package / "results"
    stage1 = pd.read_csv(result_dir / "stage1_predictions.csv")
    stage1["customer_id"] = stage1["customer_id"].astype(str)
    stage1 = stage1.sort_values(["predicted_probability", "customer_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    stage1["rank"] = np.arange(1, len(stage1)+1)
    stage1["is_top_10"] = stage1["rank"] <= 10
    stage1.to_csv(repo / "stage1_predictions.csv", index=False)

    stage2 = _json(result_dir / "stage2_selection.json")
    stage2_rows = pd.DataFrame(stage2["candidate_rows"])
    selected_ids = [str(x) for x in stage2["exact"]["selected_ids"]]
    stage2_rows["selected"] = stage2_rows["customer_id"].astype(str).isin(selected_ids)
    stage2_rows.to_csv(repo / "stage2_selection.csv", index=False)
    qaoa = stage2["qaoa"]
    exact_value = float(stage2["exact"]["objective_value"])
    qaoa_value = float(qaoa["best_sampled_objective"])
    stage2.update({
        "top_10_candidate_ids": [str(x) for x in stage2["candidate_ids_in_rank_order"]],
        "selected_six_ids": selected_ids,
        "budget_used": int(stage2["exact"]["contact_cost_used"]), "budget_limit": int(stage2["exact"]["contact_cost_limit"]),
        "capacity_used": int(stage2["exact"]["capacity_used"]), "capacity_limit": int(stage2["exact"]["capacity_limit"]),
        "objective_value": exact_value,
        "measured_best_bitstrings_with_counts": qaoa["top_10_sampled_bitstrings"],
        "classical_brute_force_optimal_objective": exact_value,
        "qaoa_best_sampled_objective": qaoa_value,
        "qaoa_optimality_gap_absolute": float(exact_value-qaoa_value),
        "qaoa_optimality_gap_relative": float((exact_value-qaoa_value)/abs(exact_value)),
        "submitted_answer_optimality_gap_absolute": 0.0,
        "submitted_answer_optimality_gap_relative": 0.0,
    })
    stage2["transpilation_diagnostics"] = _stage2_transpilation(stage2, optimization_level, seed_transpiler)
    _write_json(repo / "stage2_selection.json", stage2)

    stage3 = _json(result_dir / "stage3_simulation.json")
    ids = [str(r["customer_id"]) for r in stage3["qubit_order"]]
    probabilities = np.asarray([float(r["predicted_probability"]) for r in stage3["qubit_order"]])
    exact = float(stage3["exact_probability_from_same_six_probabilities"])
    circuit_estimate = float(stage3["estimated_success_probability"])
    circuit_error = abs(circuit_estimate-exact)
    rng = np.random.default_rng(42)
    draws = rng.binomial(1, probabilities, size=(int(mc_draws), len(probabilities)))
    successes = int(np.count_nonzero(draws.sum(axis=1) >= 3))
    mc_estimate = successes / int(mc_draws)
    stage3.update({
        "selected_customer_ids": ids,
        "circuit_parameters": {"per_qubit_gate": "RY(2*arcsin(sqrt(p)))", "angles_radians": [float(2*np.arcsin(np.sqrt(p))) for p in probabilities],
                                "classical_register_mapping": "qubit i measured to classical bit i; printed Qiskit bitstrings reverse this order"},
        "circuit_estimate_absolute_error": float(circuit_error),
        "circuit_estimate_relative_error": float(circuit_error/exact),
        "monte_carlo_benchmark": {"draws": int(mc_draws), "seed": 42, "independent_bernoulli_draws": True,
            "success_count": successes, "estimated_probability": float(mc_estimate), "wilson_95_ci": _wilson(successes, int(mc_draws)),
            "absolute_error_vs_exact": float(abs(mc_estimate-exact)), "relative_error_vs_exact": float(abs(mc_estimate-exact)/exact)},
    })
    linear = [[i, i+1] for i in range(5)] + [[i+1, i] for i in range(5)]
    noiseless_backend = AerSimulator(coupling_map=linear, basis_gates=["rz", "sx", "x", "cx"])
    _, stage3["transpilation_diagnostics"] = _transpile_stats(
        _stage3_circuit(probabilities), noiseless_backend, optimization_level, seed_transpiler)
    stage3["noise_model_comparison"] = _noise_run(probabilities, 42, noise_shots, optimization_level)
    calibration_path = result_dir / "stage1_calibration_curve.csv"
    selected_frame = stage2_rows.loc[stage2_rows["selected"]].reset_index(drop=True)
    stage3["stage1_uncertainty_propagation"] = _uncertainty_sensitivity(
        stage2_rows, calibration_path, selected_frame, exact, stage2["exact"]["contact_cost_limit"], stage2["exact"]["capacity_limit"], 42, uncertainty_draws)
    _write_json(repo / "stage3_simulation.json", stage3)
    pd.DataFrame([{**row, "stage3_quantum_estimate": circuit_estimate,
                   "stage3_exact_probability": exact, "stage3_mc_estimate": mc_estimate}
                  for row in stage3["qubit_order"]]).to_csv(repo / "stage3_simulation.csv", index=False)

    # Keep the package's canonical Stage 1/2 CSV schemas while adding Stage 3 diagnostics.
    stage1[["customer_id", "predicted_probability"]].to_csv(package / "results" / "stage1_predictions.csv", index=False)
    stage2_rows.loc[stage2_rows["selected"]].drop(columns=["selected"]).to_csv(package / "results" / "stage2_selection.csv", index=False)
    for name in ("stage2_selection.json", "stage3_simulation.csv", "stage3_simulation.json"):
        (package / "results" / name).write_bytes((repo / name).read_bytes())
    return {"root": str(repo), "mc": stage3["monte_carlo_benchmark"],
            "noise": stage3["noise_model_comparison"],
            "stage2_transpilation": stage2["transpilation_diagnostics"],
            "stage3_transpilation": stage3["transpilation_diagnostics"],
            "uncertainty": stage3["stage1_uncertainty_propagation"]}


if __name__ == "__main__":
    print(json.dumps(export_submission_artifacts(), indent=2))
