import unittest

import numpy as np

from src.eval_extras.metrics import multiclass_brier, precision_recall_at_k, pr_auc


class TestEvaluationExtraMetrics(unittest.TestCase):
    def test_pr_auc_perfect_ranking(self):
        self.assertAlmostEqual(pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_pr_auc_rejects_single_class(self):
        with self.assertRaises(ValueError):
            pr_auc([0, 0], [0.1, 0.2])

    def test_multiclass_brier_formula(self):
        got = multiclass_brier([0, 1], [[0.8, 0.2], [0.25, 0.75]])
        expected = ((0.2**2 + 0.2**2) + (0.25**2 + 0.25**2)) / 2
        self.assertAlmostEqual(got, expected)

    def test_precision_recall_at_k(self):
        got = precision_recall_at_k([0, 1, 0, 1], [0.4, 0.9, 0.8, 0.1], 2)
        self.assertEqual(got, {"precision": 0.5, "recall": 0.5, "true_positives": 1})

    def test_precision_recall_at_k_validates_k(self):
        with self.assertRaises(ValueError):
            precision_recall_at_k([0, 1], [0.1, 0.2], 3)


if __name__ == "__main__":
    unittest.main()
