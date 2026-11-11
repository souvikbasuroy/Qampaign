import shutil
import tempfile
import unittest
from pathlib import Path

import openpyxl
import numpy as np
import pandas as pd

from src.customer_pipeline.stage2 import (
    _qaoa_circuit, _vectorized_qaoa_objective, decode_bits_to_ids,
    exact_qubo_minimum, subset_resource_diagnostics, validate_selection,
)
from src.customer_pipeline.stage3 import counts_to_qubit_bits, exact_at_least_k
from qiskit.quantum_info import Statevector
from src.customer_pipeline.workbook import load_participant_workbook


WORKBOOK = Path("data/bank_campaign_participant_release.xlsx")


class WorkbookPipelineTests(unittest.TestCase):
    def test_stage2_and_stage3_match_stage1_prediction_ids_and_probabilities(self):
        root = Path("results")
        stage1 = pd.read_csv(root / "stage1_predictions.csv")
        stage2 = pd.read_csv(root / "stage2_selection.csv")
        with (root / "stage3_simulation.json").open(encoding="utf-8") as handle:
            stage3 = __import__("json").load(handle)
        probability_by_id = dict(zip(stage1.customer_id.astype(str), stage1.predicted_probability))
        selected_ids = stage2.customer_id.astype(str).tolist()
        self.assertEqual(len(selected_ids), 6)
        self.assertEqual(len(set(selected_ids)), 6)
        self.assertTrue(set(selected_ids).issubset(probability_by_id))
        for row in stage2.itertuples():
            self.assertAlmostEqual(float(row.predicted_probability), probability_by_id[str(row.customer_id)], places=12)
        stage3_ids = [str(row["customer_id"]) for row in stage3["qubit_order"]]
        self.assertEqual(stage3_ids, selected_ids)
        for row in stage3["qubit_order"]:
            self.assertAlmostEqual(float(row["predicted_probability"]), probability_by_id[row["customer_id"]], places=12)

    def test_official_workbook_ids_labels_and_constraints(self):
        data = load_participant_workbook(WORKBOOK)
        self.assertEqual((len(data["train"]), len(data["test"])), (300, 100))
        self.assertNotIn("y", data["test"].columns)
        self.assertEqual(data["mapping_status"], "WORKBOOK_IDS")
        self.assertEqual(len(set(data["test"].customer_id)), 100)
        self.assertFalse(set(data["train"].customer_id) & set(data["test"].customer_id))
        self.assertEqual(data["constraints"], {"N": 10, "K": 6, "C": 25, "H": 22})

    def test_provisional_mapping_is_explicit_and_warned(self):
        with tempfile.TemporaryDirectory() as directory:
            copy_path = Path(directory) / "without_test_ids.xlsx"
            workbook = openpyxl.load_workbook(WORKBOOK)
            sheet = workbook["Stage1_Test"]
            for row in range(2, sheet.max_row + 1):
                sheet.cell(row=row, column=1).value = None
            workbook.save(copy_path)
            workbook.close()
            data = load_participant_workbook(copy_path)
            self.assertEqual(data["mapping_status"], "PROVISIONAL_ROW_ORDER")
            self.assertIn("PROVISIONAL", data["mapping_note"])
            expected = sorted(
                set(data["stage2_data"].customer_id) - set(data["train"].customer_id)
            )
            self.assertEqual(data["test"].customer_id.tolist(), expected)

    def test_mapping_file_is_used_and_validated(self):
        data = load_participant_workbook(WORKBOOK)
        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "id_map.csv"
            pd.DataFrame({
                "test_row": np.arange(1, 101),
                "customer_id": data["test"].customer_id,
            }).to_csv(mapping_path, index=False)
            mapped = load_participant_workbook(WORKBOOK, mapping_path)
            self.assertEqual(mapped["mapping_status"], "EXPLICIT_MAPPING_FILE")
            self.assertEqual(mapped["test"].customer_id.tolist(), data["test"].customer_id.tolist())

    def test_selection_requires_six_unique_candidates_and_respects_limits(self):
        candidates = pd.DataFrame({
            "customer_id": [f"C{i:04}" for i in range(10)],
            "contact_cost": [1] * 10,
            "capacity_units": [2] * 10,
        })
        limits = {"N": 10, "K": 6, "C": 25, "H": 22}
        totals = validate_selection(candidates, candidates.customer_id[:6], limits)
        self.assertEqual(totals, {"contact_cost": 6, "capacity_units": 12})
        with self.assertRaises(ValueError):
            validate_selection(candidates, candidates.customer_id[:5], limits)
        with self.assertRaises(ValueError):
            validate_selection(candidates, ["C0000"] * 6, limits)

    def test_dicke_xy_qaoa_stays_in_weight_six_and_vectorized_cost_matches_circuit(self):
        net = np.linspace(0.1, 1.0, 10)
        p = 1
        gamma, beta = np.asarray([0.2]), np.asarray([0.35])
        circuit = _qaoa_circuit(net, p, gamma, beta)
        state = Statevector.from_instruction(circuit).data
        probabilities = np.abs(state) ** 2
        self.assertAlmostEqual(sum(probabilities[i] for i in range(1 << 10)
                                   if i.bit_count() == 6), 1.0, places=10)
        self.assertAlmostEqual(sum(probabilities[i] for i in range(1 << 10)
                                   if i.bit_count() != 6), 0.0, places=10)
        bits = np.asarray([[(idx >> q) & 1 for q in range(10)] for idx in range(1 << 10)])
        rewards = bits @ (net / np.max(np.abs(net)))
        pairs = []
        for left in range(10):
            for right in range(left + 1, 10):
                idx01 = np.asarray([idx for idx in range(1 << 10)
                                    if ((idx >> left) & 1) == 0 and ((idx >> right) & 1) == 1])
                pairs.append((idx01, idx01 ^ (1 << left) ^ (1 << right)))
        fast = _vectorized_qaoa_objective(np.r_[gamma, beta], p,
                                          net / np.max(np.abs(net)), rewards, pairs)
        self.assertAlmostEqual(fast, -float(np.dot(probabilities, rewards)), places=10)

    def test_resource_constraints_are_checked_generally_for_every_subset(self):
        frame = pd.DataFrame({"contact_cost": [1] * 10, "capacity_units": [1] * 10})
        inactive = subset_resource_diagnostics(frame, 6, 25, 22)
        self.assertEqual(inactive["total_count"], 210)
        self.assertTrue(inactive["all_feasible"])
        self.assertEqual((inactive["max_cost"], inactive["max_capacity"]), (6, 6))
        active = subset_resource_diagnostics(frame, 6, 5, 22)
        self.assertFalse(active["all_feasible"])
        self.assertEqual(active["feasible_count"], 0)

    def test_qaoa_bitstring_decodes_to_ranked_customer_ids(self):
        candidates = pd.DataFrame({"customer_id": [f"C{i:04}" for i in range(10)]})
        decoded = decode_bits_to_ids("1010010001", candidates)
        self.assertEqual(decoded, ["C0000", "C0002", "C0005", "C0009"])

    def test_penalty_qubo_minimum_matches_brute_force_optimum(self):
        net = np.asarray([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        costs = np.asarray([2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
        capacities = np.asarray([2, 2, 3, 2, 3, 2, 2, 3, 2, 3])
        budget, capacity_limit, k = 16, 15, 6
        feasible = []
        from itertools import combinations
        for chosen in combinations(range(10), k):
            if costs[list(chosen)].sum() <= budget and capacities[list(chosen)].sum() <= capacity_limit:
                feasible.append(float(net[list(chosen)].sum()))
        exact_objective = max(feasible)
        self.assertAlmostEqual(
            exact_qubo_minimum(net, costs, capacities, budget, capacity_limit, k),
            -exact_objective,
        )

    def test_qiskit_little_endian_bit_decoding(self):
        self.assertEqual(counts_to_qubit_bits("000001"), (1, 0, 0, 0, 0, 0))
        self.assertEqual(counts_to_qubit_bits("100000"), (0, 0, 0, 0, 0, 1))

    def test_exact_independent_probability(self):
        self.assertAlmostEqual(exact_at_least_k([0.5] * 6, 3), 42 / 64)



    def test_submission_root_artifacts_have_required_schemas_and_match_package_results(self):
        repo = Path(__file__).resolve().parents[1]
        package_results = Path(__file__).resolve().parents[1] / "results"
        stage1 = pd.read_csv(repo / "stage1_predictions.csv")
        self.assertEqual(len(stage1), 100)
        self.assertTrue({"customer_id", "predicted_probability", "rank", "is_top_10"}.issubset(stage1.columns))
        self.assertEqual(stage1["rank"].tolist(), list(range(1, 101)))
        expected = stage1.sort_values(["predicted_probability", "customer_id"], ascending=[False, True], kind="mergesort")
        self.assertEqual(stage1["customer_id"].tolist(), expected["customer_id"].tolist())
        stage2 = pd.read_csv(repo / "stage2_selection.csv")
        stage2_json = __import__("json").loads((repo / "stage2_selection.json").read_text(encoding="utf-8"))
        self.assertTrue({"customer_id", "predicted_probability", "selected"}.issubset(stage2.columns))
        self.assertEqual(int(stage2["selected"].sum()), 6)
        for key in ("top_10_candidate_ids", "selected_six_ids", "budget_used", "budget_limit",
                    "capacity_used", "capacity_limit", "objective_value", "measured_best_bitstrings_with_counts",
                    "classical_brute_force_optimal_objective", "qaoa_optimality_gap_relative",
                    "submitted_answer_optimality_gap_relative", "transpilation_diagnostics"):
            self.assertIn(key, stage2_json)
        package_stage2 = __import__("json").loads((package_results / "stage2_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(stage2_json["selected_six_ids"], package_stage2["exact"]["selected_ids"])
        stage3 = __import__("json").loads((repo / "stage3_simulation.json").read_text(encoding="utf-8"))
        stage3_csv = pd.read_csv(repo / "stage3_simulation.csv")
        for key in ("selected_customer_ids", "qubit_order", "circuit_parameters", "counts",
                    "estimated_success_probability", "exact_probability_from_same_six_probabilities",
                    "monte_carlo_benchmark", "noise_model_comparison", "transpilation_diagnostics",
                    "stage1_uncertainty_propagation"):
            self.assertIn(key, stage3)
        self.assertEqual(len(stage3_csv), 6)
        self.assertEqual(stage3["selected_customer_ids"], stage2_json["selected_six_ids"])
        self.assertEqual(stage3["monte_carlo_benchmark"]["seed"], 42)
        self.assertEqual(stage3["transpilation_diagnostics"]["physical_qubits"], 6)

if __name__ == "__main__":
    unittest.main()
