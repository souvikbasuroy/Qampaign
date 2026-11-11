"""Stage 3: independent Bernoulli shot simulation and exact verification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.stats import binomtest


def counts_to_qubit_bits(bitstring):
    """Qiskit count strings print classical bit 0 on the right."""
    return tuple(int(character) for character in str(bitstring).replace(" ", "")[::-1])


def exact_at_least_k(probabilities, k=3):
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    distribution = np.zeros(len(probabilities) + 1, dtype=float)
    distribution[0] = 1.0
    used = 0
    for probability in probabilities:
        distribution[1:used + 2] = (
            distribution[1:used + 2] * (1 - probability)
            + distribution[:used + 1] * probability
        )
        distribution[0] *= 1 - probability
        used += 1
    return float(distribution[int(k):].sum())


def run_simulation(selected, cfg, seed, out_dir, id_mapping_status, id_mapping_note):
    selected = selected.reset_index(drop=True)
    probabilities = selected["predicted_probability"].to_numpy(float)
    if len(selected) != 6 or selected["customer_id"].nunique() != 6:
        raise ValueError("Stage 3 requires exactly six unique selected customers.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Selected probabilities must lie in [0, 1].")

    circuit = QuantumCircuit(6, 6)
    for qubit, probability in enumerate(probabilities):
        circuit.ry(2.0 * np.arcsin(np.sqrt(probability)), qubit)
        circuit.measure(qubit, qubit)
    backend = AerSimulator()
    compiled = transpile(circuit, backend, seed_transpiler=int(seed))
    shots = int(cfg["shots"])
    counts = backend.run(
        compiled, shots=shots, seed_simulator=int(seed)
    ).result().get_counts()
    success_shots = sum(
        count for bitstring, count in counts.items()
        if sum(counts_to_qubit_bits(bitstring)) >= 3
    )
    estimate = success_shots / shots
    interval = binomtest(success_shots, shots).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    exact = exact_at_least_k(probabilities, k=3)
    order = [
        {
            "qubit": int(index),
            "customer_id": str(row.customer_id),
            "predicted_probability": float(row.predicted_probability),
        }
        for index, row in enumerate(selected.itertuples())
    ]
    sensitivity = [
        {"scenario": f"each_customer_p_{rate:.2f}", "per_customer_probability": rate,
         "probability_at_least_3_of_6": exact_at_least_k([rate] * 6, k=3)}
        for rate in (0.10, 0.15, 0.20, 0.25)
    ]
    sensitivity.append({
        "scenario": "submitted_six_probabilities",
        "per_customer_probability": None,
        "probability_at_least_3_of_6": exact,
    })
    result = {
        "id_mapping_status": id_mapping_status,
        "id_mapping_warning": id_mapping_note,
        "success_definition": "at least 3 of 6 measured qubits are 1",
        "qubit_order": order,
        "bit_order_note": "Qiskit count strings are little-endian: qubit 0 is the rightmost character.",
        "shots": shots,
        "seed": int(seed),
        "backend": "Qiskit AerSimulator (ideal statevector shot sampling)",
        "qubit_count": 6,
        "counts": {str(key): int(value) for key, value in counts.items()},
        "success_shots": int(success_shots),
        "estimated_success_probability": float(estimate),
        "estimated_success_ci95": [float(interval.low), float(interval.high)],
        "exact_probability_from_same_six_probabilities": exact,
        "sensitivity_table": sensitivity,
        "absolute_sampling_error": float(abs(estimate - exact)),
    }
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage3_simulation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    import pandas as pd
    pd.DataFrame(sensitivity).to_csv(output_dir / "stage3_sensitivity.csv", index=False)
    return result
