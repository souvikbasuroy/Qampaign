"""Stage 2: workbook economics, exact selection, QUBO/QAOA comparison."""
from __future__ import annotations

import json
from math import comb
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def qubo_energy_for_selection(bits, net_values, costs, capacities, budget, capacity_limit,
                              selected_count=6, penalty=10000.0):
    """Minimum QUBO energy over binary slack assignments for one decision bitstring."""
    bits = np.asarray(bits, dtype=int)
    chosen = bits.astype(bool)
    budget_used = int(np.dot(bits, costs))
    capacity_used = int(np.dot(bits, capacities))
    count = int(bits.sum())
    violation = (
        max(0, count - int(selected_count)) ** 2
        + max(0, int(selected_count) - count) ** 2
        + max(0, budget_used - int(budget)) ** 2
        + max(0, capacity_used - int(capacity_limit)) ** 2
    )
    return float(-np.dot(bits, net_values) + float(penalty) * violation)


def _build_qubo(net_values, costs, capacities, budget, capacity_limit,
                selected_count, penalty):
    n = len(net_values)
    qp = QuadraticProgram("workbook_customer_selection")
    decision_names = [f"x{i}" for i in range(n)]
    budget_slack_count = max(1, int(np.ceil(np.log2(int(budget) + 1))))
    capacity_slack_count = max(1, int(np.ceil(np.log2(int(capacity_limit) + 1))))
    for name in decision_names:
        qp.binary_var(name=name)
    budget_slack = [f"budget_slack_{i}" for i in range(budget_slack_count)]
    capacity_slack = [f"capacity_slack_{i}" for i in range(capacity_slack_count)]
    for name in budget_slack + capacity_slack:
        qp.binary_var(name=name)

    linear = {decision_names[i]: -float(net_values[i]) for i in range(n)}
    quadratic = {}

    def add_square(weights, target):
        active = [(name, float(weight)) for name, weight in weights.items() if weight]
        constant = -float(target)
        for name, weight in active:
            linear[name] = linear.get(name, 0.0) + penalty * (2 * constant * weight + weight * weight)
        for index, (name_i, weight_i) in enumerate(active):
            for name_j, weight_j in active[index + 1:]:
                key = (name_i, name_j)
                quadratic[key] = quadratic.get(key, 0.0) + 2 * penalty * weight_i * weight_j

    add_square({name: 1 for name in decision_names}, selected_count)
    budget_weights = {decision_names[i]: costs[i] for i in range(n)}
    budget_weights.update({name: 2 ** i for i, name in enumerate(budget_slack)})
    add_square(budget_weights, budget)
    capacity_weights = {decision_names[i]: capacities[i] for i in range(n)}
    capacity_weights.update({name: 2 ** i for i, name in enumerate(capacity_slack)})
    add_square(capacity_weights, capacity_limit)
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp, budget_slack_count, capacity_slack_count



def subset_resource_diagnostics(candidates, selected_count, budget, capacity_limit):
    totals = [
        (int(candidates.iloc[list(subset)]["contact_cost"].sum()),
         int(candidates.iloc[list(subset)]["capacity_units"].sum()))
        for subset in combinations(range(len(candidates)), int(selected_count))
    ]
    feasible = [cost <= int(budget) and capacity <= int(capacity_limit)
                for cost, capacity in totals]
    return {
        "all_feasible": bool(all(feasible)),
        "feasible_count": int(sum(feasible)), "total_count": len(totals),
        "max_cost": max(cost for cost, _ in totals),
        "max_capacity": max(capacity for _, capacity in totals),
        "limits_inactive": bool(all(feasible)),
    }


def validate_selection(candidates, selected_ids, constraints):
    ids = [str(value) for value in selected_ids]
    if len(ids) != int(constraints["K"]) or len(set(ids)) != len(ids):
        raise ValueError("Selection must contain exactly K unique customer IDs.")
    candidate_ids = set(candidates["customer_id"].astype(str))
    if not set(ids).issubset(candidate_ids):
        raise ValueError("Selected IDs must all belong to the candidate pool.")
    selected = candidates.set_index(candidates["customer_id"].astype(str)).loc[ids]
    cost_total = int(selected["contact_cost"].sum())
    capacity_total = int(selected["capacity_units"].sum())
    if cost_total > int(constraints["C"]) or capacity_total > int(constraints["H"]):
        raise ValueError("Selection exceeds a workbook budget or capacity constraint.")
    return {"contact_cost": cost_total, "capacity_units": capacity_total}


def decode_bits_to_ids(bitstring, candidates):
    """Decode left-to-right customer decision bits in top-10 rank order."""
    bits = str(bitstring).replace(" ", "")
    if len(bits) != len(candidates) or set(bits).difference("01"):
        raise ValueError("Decision bitstring must contain one binary bit per candidate.")
    return [str(cid) for bit, cid in zip(bits, candidates["customer_id"]) if bit == "1"]


def _qaoa_circuit(net_values, p, gammas, betas, measure=False):
    n = len(net_values)
    k = 6
    states = [index for index in range(1 << n) if index.bit_count() == k]
    initial = np.zeros(1 << n, dtype=complex)
    initial[states] = 1 / np.sqrt(len(states))
    circuit = QuantumCircuit(n, n if measure else 0)
    circuit.append(StatePreparation(initial), range(n))
    scale = max(float(np.max(np.abs(net_values))), 1e-12)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for layer in range(p):
        for qubit, value in enumerate(net_values):
            circuit.rz(-2 * float(gammas[layer]) * float(value) / scale, qubit)
        # Complete-graph XY mixer preserves Hamming weight and connects the weight-6 sector.
        for left, right in edges:
            circuit.rxx(float(betas[layer]), left, right)
            circuit.ryy(float(betas[layer]), left, right)
    if measure:
        circuit.measure(range(n), range(n))
    return circuit


def _spsa(objective, initial, maxiter, rng):
    point = np.asarray(initial, dtype=float).copy()
    best_point = point.copy()
    best_value = float(objective(point))
    history = [best_value]
    for step in range(maxiter):
        a = 0.15 / ((step + 1) ** 0.602)
        c = 0.12 / ((step + 1) ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=len(point))
        plus = float(objective(point + c * delta))
        minus = float(objective(point - c * delta))
        gradient = ((plus - minus) / (2 * c)) * delta
        point = point - a * gradient
        current = float(objective(point))
        if current < best_value:
            best_value, best_point = current, point.copy()
        history.append(best_value)
    return best_point, best_value, history


def _vectorized_qaoa_objective(theta, p, normalized, rewards, pair_indices):
    """NumPy statevector evolution; exactly applies the same RZ then XX/YY pair gates."""
    n = len(normalized)
    state = np.zeros(1 << n, dtype=complex)
    weight_six = [index for index in range(1 << n) if index.bit_count() == 6]
    state[weight_six] = 1 / np.sqrt(len(weight_six))
    gammas = theta[:p]
    betas = theta[p:]
    phase = np.exp(-2j * gammas[0] * rewards) if p else 1
    for layer in range(p):
        if layer:
            phase = np.exp(-2j * gammas[layer] * rewards)
        state *= phase
        cosine, sine = np.cos(betas[layer]), np.sin(betas[layer])
        for idx01, idx10 in pair_indices:
            a, b = state[idx01].copy(), state[idx10].copy()
            state[idx01] = cosine * a - 1j * sine * b
            state[idx10] = cosine * b - 1j * sine * a
    probabilities = np.abs(state) ** 2
    return -float(np.dot(probabilities, rewards))


def _run_dicke_qaoa(net_values, costs, capacities, budget, capacity_limit, cfg, seed,
                    exact_bitstring):
    p_values = [int(value) for value in cfg["qaoa_p_values"]]
    restarts = int(cfg["qaoa_restarts"])
    maxiter = int(cfg["qaoa_maxiter"])
    shots = int(cfg["qaoa_shots"])
    if restarts < 10 or maxiter < 200 or shots < 8192:
        raise ValueError("Dicke QAOA requires at least 10 restarts, 200 iterations, and 8192 shots.")
    n, k = len(net_values), 6
    size = 1 << n
    basis_bits = np.asarray([[(index >> qubit) & 1 for qubit in range(n)] for index in range(size)])
    normalized = np.asarray(net_values, dtype=float) / max(float(np.max(np.abs(net_values))), 1e-12)
    rewards = basis_bits @ normalized
    pairs = []
    for left in range(n):
        for right in range(left + 1, n):
            idx01 = np.asarray([idx for idx in range(size) if ((idx >> left) & 1) == 0 and ((idx >> right) & 1) == 1])
            idx10 = idx01 ^ (1 << left) ^ (1 << right)
            pairs.append((idx01, idx10))
    records = []
    best_run = None
    for p in p_values:
        for optimizer_name in ("COBYLA", "SPSA"):
            for restart in range(restarts):
                run_seed = int(seed + p * 10000 + (0 if optimizer_name == "COBYLA" else 5000) + restart)
                rng = np.random.default_rng(run_seed)
                initial = rng.uniform(-np.pi, np.pi, size=2 * p)

                def objective(theta):
                    return _vectorized_qaoa_objective(theta, p, normalized, rewards, pairs)

                if optimizer_name == "COBYLA":
                    evaluations = []
                    def tracked(theta):
                        value = objective(theta)
                        evaluations.append(-value)
                        return value
                    fitted = minimize(
                        tracked, initial, method="COBYLA",
                        options={"maxiter": maxiter, "rhobeg": 0.25, "tol": 1e-5},
                    )
                    theta = np.asarray(fitted.x, dtype=float)
                    value = -float(objective(theta))
                    history = np.maximum.accumulate(np.asarray(evaluations, dtype=float)).tolist()
                else:
                    theta, loss, history_loss = _spsa(objective, initial, maxiter, rng)
                    value = -float(loss)
                    history = (-np.asarray(history_loss, dtype=float)).tolist()
                record = {"p": p, "optimizer": optimizer_name, "restart": restart,
                          "seed": run_seed, "best_expected_normalized_objective": value,
                          "iterations": maxiter, "convergence": history,
                          "parameters": theta.tolist()}
                records.append(record)
                if best_run is None or value > best_run["best_expected_normalized_objective"]:
                    best_run = record
    backend = AerSimulator()
    measured_runs = []
    winner_counts = None
    winner = None
    for record in records:
        circuit = _qaoa_circuit(
            net_values, record["p"], record["parameters"][:record["p"]],
            record["parameters"][record["p"]:], measure=True,
        )
        compiled = transpile(circuit, backend, seed_transpiler=int(record["seed"]))
        counts_i = backend.run(
            compiled, shots=shots, seed_simulator=int(seed)
        ).result().get_counts()
        optimum_count = sum(
            count for bitstring, count in counts_i.items()
            if bitstring.replace(" ", "")[::-1] == exact_bitstring
        )
        feasible_count = 0
        best_sample_value = None
        for bitstring, count in counts_i.items():
            decision = bitstring.replace(" ", "")[::-1]
            bits = np.asarray([int(bit) for bit in decision], dtype=int)
            feasible = (int(bits.sum()) == k and int(np.dot(bits, costs)) <= budget
                        and int(np.dot(bits, capacities)) <= capacity_limit)
            if feasible:
                feasible_count += int(count)
                value = float(np.dot(bits, net_values))
                best_sample_value = value if best_sample_value is None else max(best_sample_value, value)
        record["sampled_optimum_probability_mass"] = optimum_count / shots
        record["sampled_feasibility_rate"] = feasible_count / shots
        record["best_feasible_sampled_objective"] = best_sample_value
        measured_runs.append(record)
        if winner is None or (
            record["sampled_optimum_probability_mass"],
            record["best_expected_normalized_objective"],
        ) > (
            winner["sampled_optimum_probability_mass"],
            winner["best_expected_normalized_objective"],
        ):
            winner = record
            winner_counts = counts_i
    best_run = winner
    counts = winner_counts
    ranked_counts = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    ranked_counts = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    top_samples, feasible_shots, optimum_shots = [], 0, 0
    sampled_best = None
    for bitstring, count in ranked_counts:
        decision = bitstring.replace(" ", "")[::-1]
        bit_vector = np.asarray([int(bit) for bit in decision], dtype=int)
        cost_total, capacity_total = int(np.dot(bit_vector, costs)), int(np.dot(bit_vector, capacities))
        value = float(np.dot(bit_vector, net_values))
        feasible = int(bit_vector.sum()) == k and cost_total <= budget and capacity_total <= capacity_limit
        if feasible:
            feasible_shots += int(count)
            if sampled_best is None or value > sampled_best["objective"]:
                sampled_best = {"bitstring": decision, "objective": value,
                                "contact_cost": cost_total, "capacity_units": capacity_total,
                                "rank_indices": np.flatnonzero(bit_vector).tolist()}
        if decision == exact_bitstring:
            optimum_shots += int(count)
        top_samples.append({"bitstring_x0_to_xN": decision,
                            "selected_rank_indices": np.flatnonzero(bit_vector).tolist(),
                            "selected_count": int(bit_vector.sum()), "objective_value": value,
                            "contact_cost": cost_total, "capacity_units": capacity_total,
                            "feasible": bool(feasible), "count": int(count)})
    top_samples = top_samples[:10]
    return {
        "backend": "Qiskit AerSimulator for final shots; vectorized exact statevector objective during optimization",
        "seed": int(seed), "shots": shots, "restarts_per_setting": restarts,
        "p_values": p_values, "optimizers": ["COBYLA", "SPSA"], "maxiter": maxiter,
        "normalization": "net value coefficients divided by max(abs(net value coefficient))",
        "initial_state": "uniform Dicke state over all weight-6 bitstrings",
        "mixer": "complete-graph XY mixer using sequential RXX(beta) then RYY(beta) on each pair; preserves Hamming weight",
        "constraint_handling": "decode decision bits, then post-select/check K, workbook budget, and workbook capacity; no slack qubits",
        "decision_qubits": n, "feasibility_rate": feasible_shots / shots,
        "exact_optimum_probability_mass": optimum_shots / shots,
        "exact_optimum_random_chance": 1 / comb(n, k),
        "best_of_80_measured_runs": {
            "runs": len(measured_runs),
            "median_optimum_probability_mass": float(np.median([row["sampled_optimum_probability_mass"] for row in measured_runs])),
            "mean_optimum_probability_mass": float(np.mean([row["sampled_optimum_probability_mass"] for row in measured_runs])),
            "threshold_definition": "at least 10% optimum probability mass per 8192-shot run",
            "threshold_value": 0.10,
            "runs_at_or_above_threshold": int(sum(row["sampled_optimum_probability_mass"] >= 0.10 for row in measured_runs)),
            "best_run_settings": {key: best_run[key] for key in ("p", "optimizer", "restart", "seed", "best_expected_normalized_objective", "sampled_optimum_probability_mass", "sampled_feasibility_rate")},
        },
        "all_run_sample_metrics": [{key: row[key] for key in ("p", "optimizer", "restart", "seed", "best_expected_normalized_objective", "sampled_optimum_probability_mass", "sampled_feasibility_rate", "best_feasible_sampled_objective")} for row in measured_runs],
        "best_feasible_sample": sampled_best,
        "best_sampled_objective": sampled_best["objective"] if sampled_best else None,
        "top_10_sampled_bitstrings": top_samples,
        "best_optimization_run": {key: value for key, value in best_run.items() if key != "convergence"},
        "highest_expected_objective_run": {key: value for key, value in max(measured_runs, key=lambda row: row["best_expected_normalized_objective"]).items() if key != "convergence"},
        "best_convergence": best_run["convergence"],
        "sweep_summary": [{key: value for key, value in row.items() if key not in {"parameters", "convergence"}} for row in records],
        "all_run_convergence": [row["convergence"] for row in records],
        "sample_counts": {str(bitstring): int(count) for bitstring, count in counts.items()},
    }

def solve_selection(top10, stage2_data, constraints, cfg, seed, out_dir,
                    id_mapping_status, id_mapping_note, robustness_candidate_pool=None):
    n, k = int(constraints["N"]), int(constraints["K"])
    budget, capacity_limit = int(constraints["C"]), int(constraints["H"])
    candidates = top10.head(n).merge(
        stage2_data, on="customer_id", how="left", validate="one_to_one", sort=False
    )
    if len(candidates) != n or candidates[["contact_cost", "expected_deposit_value", "capacity_units"]].isna().any().any():
        raise ValueError("Top candidates did not all join to Stage2_Data.")
    candidates["contact_cost"] = pd.to_numeric(candidates["contact_cost"], errors="raise")
    candidates["expected_deposit_value"] = pd.to_numeric(candidates["expected_deposit_value"], errors="raise")
    candidates["capacity_units"] = pd.to_numeric(candidates["capacity_units"], errors="raise")
    candidates["expected_net_value"] = (
        candidates["predicted_probability"] * candidates["expected_deposit_value"]
        - candidates["contact_cost"]
    )
    net = candidates["expected_net_value"].to_numpy(float)
    costs = candidates["contact_cost"].to_numpy(int)
    capacities = candidates["capacity_units"].to_numpy(int)

    feasible = []
    for subset in combinations(range(n), k):
        chosen = candidates.iloc[list(subset)]
        cost_total = int(chosen["contact_cost"].sum())
        capacity_total = int(chosen["capacity_units"].sum())
        if cost_total <= budget and capacity_total <= capacity_limit:
            feasible.append({
                "indices": list(subset),
                "objective": float(chosen["expected_net_value"].sum()),
                "contact_cost": cost_total,
                "capacity_units": capacity_total,
            })
    if not feasible:
        raise ValueError("No feasible exactly-six selection under workbook constraints.")
    exact = max(feasible, key=lambda item: (item["objective"], tuple(-i for i in item["indices"])))
    exact_bits = np.zeros(n, dtype=int)
    exact_bits[exact["indices"]] = 1
    exact_bitstring = "".join(str(int(value)) for value in exact_bits)

    penalty = float(cfg["qubo_penalty"])
    qubo, budget_slack_n, capacity_slack_n = _build_qubo(
        net, costs, capacities, budget, capacity_limit, k, penalty
    )
    qaoa = _run_dicke_qaoa(
        net, costs, capacities, budget, capacity_limit, cfg, seed, exact_bitstring
    )
    qaoa["best_of_80_measured_runs"]["random_feasible_subset_chance"] = 1 / comb(n, k)
    exact_decision = exact_bitstring
    qaoa["exact_optimum_bitstring_x0_to_xN"] = exact_decision
    qaoa["exact_optimum_ids"] = selected_ids = candidates.iloc[exact["indices"]]["customer_id"].astype(str).tolist()
    for sample in qaoa["top_10_sampled_bitstrings"]:
        sample["selected_ids"] = decode_bits_to_ids(sample["bitstring_x0_to_xN"], candidates)
    qaoa["exact_optimum_probability_mass"] = sum(
        count for bitstring, count in qaoa["sample_counts"].items()
        if bitstring.replace(" ", "")[::-1] == exact_decision
    ) / int(cfg["qaoa_shots"])
    if qaoa["best_feasible_sample"] is not None:
        qaoa["best_feasible_sample"]["customer_ids"] = [
            str(candidates.iloc[i]["customer_id"])
            for i in qaoa["best_feasible_sample"]["rank_indices"]
        ]
    qaoa["approximation_ratio_to_exact"] = (
        float(qaoa["best_sampled_objective"] / exact["objective"])
        if qaoa["best_sampled_objective"] is not None and exact["objective"] > 0 else None
    )
    qaoa["submitted_selection"] = "exact feasible optimum (brute force over all subsets)"

    # Check the encoded QUBO's minimum over all decision vectors, minimizing slack analytically.
    qubo_energies = [
        qubo_energy_for_selection(bits, net, costs, capacities, budget, capacity_limit, k, penalty)
        for bits in product((0, 1), repeat=n)
    ]
    qubo_minimum = min(qubo_energies)
    qubo_matches_exact = bool(np.isclose(qubo_minimum, -exact["objective"], atol=1e-7))
    if not qubo_matches_exact:
        raise AssertionError("QUBO minimum does not agree with the exact feasible optimum.")

    selected = candidates.iloc[exact["indices"]].copy().reset_index(drop=True)
    selected["selection_bit"] = 1
    selected["qubit"] = np.arange(k, dtype=int)
    validity = validate_selection(candidates, selected["customer_id"].tolist(), constraints)
    valid = len(selected) == k
    if not valid:
        raise AssertionError("Submitted exact selection violates workbook constraints.")

    sensitivity = None
    if robustness_candidate_pool is not None and len(robustness_candidate_pool) > n:
        alternative_pool = pd.concat([
            robustness_candidate_pool.head(n - 1), robustness_candidate_pool.iloc[[n]]
        ], ignore_index=True)
        alt_candidates = alternative_pool.merge(stage2_data, on="customer_id", how="left", validate="one_to_one")
        alt_candidates["expected_net_value"] = (
            alt_candidates["predicted_probability"] * pd.to_numeric(alt_candidates["expected_deposit_value"])
            - pd.to_numeric(alt_candidates["contact_cost"])
        )
        alt_feasible = []
        for subset in combinations(range(n), k):
            chosen_alt = alt_candidates.iloc[list(subset)]
            cost_alt = int(chosen_alt["contact_cost"].sum())
            capacity_alt = int(chosen_alt["capacity_units"].sum())
            if cost_alt <= budget and capacity_alt <= capacity_limit:
                alt_feasible.append((float(chosen_alt["expected_net_value"].sum()), subset, cost_alt, capacity_alt))
        alt_best = max(alt_feasible, key=lambda item: (item[0], tuple(-i for i in item[1])))
        alt_ids = alt_candidates.iloc[list(alt_best[1])]["customer_id"].astype(str).tolist()
        sensitivity = {
            "purpose": "robustness-only rank-10/rank-11 swap; never used for model/selection choice",
            "baseline_pool_ids": candidates["customer_id"].astype(str).tolist(),
            "alternative_pool_ids": alt_candidates["customer_id"].astype(str).tolist(),
            "swapped_out_id": str(robustness_candidate_pool.iloc[n - 1]["customer_id"]),
            "swapped_in_id": str(robustness_candidate_pool.iloc[n]["customer_id"]),
            "alternative_exact_selected_ids": alt_ids,
            "alternative_objective_value": alt_best[0],
            "baseline_objective_value": float(exact["objective"]),
            "objective_change": float(alt_best[0] - exact["objective"]),
            "selected_id_overlap_count": len(set(alt_ids) & set(selected["customer_id"].astype(str))),
            "cost": alt_best[2], "capacity": alt_best[3],
        }

    output = {
        "id_mapping_status": id_mapping_status,
        "id_mapping_warning": id_mapping_note,
        "candidate_ids_in_rank_order": candidates["customer_id"].astype(str).tolist(),
        "candidate_rows": candidates[[
            "customer_id", "predicted_probability", "contact_cost",
            "expected_deposit_value", "capacity_units", "customer_segment",
            "expected_net_value",
        ]].to_dict(orient="records"),
        "objective": "sum(predicted_probability * expected_deposit_value - contact_cost)",
        "top_ten_pool_sensitivity": sensitivity,
        "feasibility_diagnostics": {
            **subset_resource_diagnostics(candidates, k, budget, capacity_limit),
            "all_six_of_ten_feasible": len(feasible) == comb(n, k),
            "feasible_combinations": len(feasible),
            "total_combinations": comb(n, k),
            "maximum_six_customer_contact_cost": max(int(candidates.iloc[list(subset)]["contact_cost"].sum()) for subset in combinations(range(n), k)),
            "maximum_six_customer_capacity": max(int(candidates.iloc[list(subset)]["capacity_units"].sum()) for subset in combinations(range(n), k)),
            "limits_inactive_for_this_top10": all(int(candidates.iloc[list(subset)]["contact_cost"].sum()) <= budget and int(candidates.iloc[list(subset)]["capacity_units"].sum()) <= capacity_limit for subset in combinations(range(n), k)),
        },
        "exact": {
            "combinations_checked": comb(n, k),
            "feasible_combinations": len(feasible),
            "selected_ids": selected["customer_id"].astype(str).tolist(),
            "bitstring_x0_to_xN": exact_bitstring,
            "decoding": "left-to-right bit i corresponds to top-10 rank i+1 and that row's customer_id",
            "objective_value": exact["objective"],
            "contact_cost_used": exact["contact_cost"],
            "contact_cost_limit": budget,
            "capacity_used": exact["capacity_units"],
            "capacity_limit": capacity_limit,
            "valid": bool(valid),
        },
        "qubo": {
            "decision_variables": n,
            "alternative_status": "20-qubit slack QUBO retained as an alternative encoding; its penalty minimum is checked against the exact subset optimum.",
            "budget_slack_variables": budget_slack_n,
            "capacity_slack_variables": capacity_slack_n,
            "penalty_weight": penalty,
            "minimum_energy_over_decision_vectors": qubo_minimum,
            "matches_exact_bruteforce_optimum": qubo_matches_exact,
            "constraint_encoding": (
                "Exactly-K uses penalty*(sum(x)-K)^2. Each <= resource limit uses "
                "penalty*(sum(resource_i*x_i)+binary_slack-limit)^2; binary slack "
                "bits represent 0 through the limit."
            ),
        },
        "qaoa": qaoa,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "stage2_selection.csv", index=False)
    (out / "stage2_selection.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(qaoa["sweep_summary"]).to_csv(out / "stage2_qaoa_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(qaoa["best_convergence"])), qaoa["best_convergence"])
    ax.set(xlabel="Objective evaluation", ylabel="Best normalized expected objective",
           title=f"Dicke/XY QAOA convergence ({qaoa['best_optimization_run']['optimizer']}, p={qaoa['best_optimization_run']['p']})")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "stage2_qaoa_convergence.png", dpi=170); plt.close(fig)
    return selected, output


def exact_qubo_minimum(net_values, costs, capacities, budget, capacity_limit,
                       selected_count, penalty=10000.0):
    """Small-instance diagnostic for tests: minimum energy over all decision bitstrings."""
    n = len(net_values)
    return min(
        qubo_energy_for_selection(bits, net_values, costs, capacities, budget,
                                  capacity_limit, selected_count, penalty)
        for bits in product((0, 1), repeat=n)
    )
