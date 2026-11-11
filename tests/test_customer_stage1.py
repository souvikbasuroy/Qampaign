import unittest

import pandas as pd

from src.customer_pipeline.stage1 import (
    CampaignFeatureEngineer, FeatureMapScale, candidate_grid, choose_cv_candidate,
    rank_predictions, validate_stage1_inputs,
)


class Stage1RulesTests(unittest.TestCase):
    def test_top10_ties_use_alphanumeric_customer_id(self):
        rows = pd.DataFrame({
            "customer_id": ["C0009", "C0002", "C0010", "C0001"],
            "predicted_probability": [0.5, 0.7, 0.5, 0.5],
        })
        ranked = rank_predictions(rows)
        self.assertEqual(ranked.customer_id.tolist(), ["C0002", "C0001", "C0009", "C0010"])

    def test_engineered_features_and_angle_range(self):
        raw = pd.DataFrame({"balance": [-99, 0], "campaign": [3, 1],
                            "pdays": [-1, 0], "previous": [0, 2],
                            "poutcome": ["unknown", "success"]})
        engineered = CampaignFeatureEngineer().fit_transform(raw)
        self.assertEqual(engineered.balance_log_signed.iloc[0], -__import__("math").log1p(99))
        self.assertEqual(engineered.previously_contacted.tolist(), [0, 1])
        self.assertEqual(engineered.previous_positive.tolist(), [0, 1])
        self.assertEqual(engineered.poutcome_success.tolist(), [0, 1])
        self.assertTrue((FeatureMapScale(0.5).transform([[0, 2 * __import__("math").pi]]) <= __import__("math").pi).all())

    def test_classical_and_quantum_families_have_equal_40_setting_budgets(self):
        cfg = {"logistic_c": [0.03, 0.1, 0.3, 1.0], "quantum_c": [0.01, 0.1, 1, 10, 100],
               "quantum_qubit_counts": [4, 6], "quantum_reps": [1, 2],
               "quantum_kernel_scales": [0.5, 1.0]}
        candidates = candidate_grid(cfg)
        self.assertEqual(sum(row["family"] == "logistic" for row in candidates), 40)
        self.assertEqual(sum(row["family"] == "quantum_zz_svc" for row in candidates), 40)

    def test_test_target_and_duration_are_rejected_before_fitting(self):
        train = pd.DataFrame({"customer_id": ["C0001"], "y": [1], "age": [40]})
        test = pd.DataFrame({"customer_id": ["C0002"], "age": [30]})
        validate_stage1_inputs(train, test)
        with self.assertRaises(ValueError):
            validate_stage1_inputs(train, test.assign(y=[0]))
        with self.assertRaises(ValueError):
            validate_stage1_inputs(train.assign(duration=[10]), test)

    def test_cv_choice_uses_only_train_cv_log_loss(self):
        cv = {
            "logistic_all": {"summary": {"log_loss": {"mean": 0.31}}},
            "quantum_zz": {"summary": {"log_loss": {"mean": 0.29}}},
        }
        self.assertEqual(choose_cv_candidate(cv), "quantum_zz")


if __name__ == "__main__":
    unittest.main()
