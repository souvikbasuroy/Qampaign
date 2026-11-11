"""Metric helpers for the optional evaluation report."""

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def pr_auc(y_true, scores):
    """Trapezoidal area under the precision-recall curve (larger = more anomalous)."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s) or not len(y) or len(np.unique(y)) != 2:
        raise ValueError("PR-AUC needs equal non-empty arrays containing both classes")
    precision, recall, _ = precision_recall_curve(y, s)
    return float(auc(recall[::-1], precision[::-1]))


def multiclass_brier(y_true, probabilities):
    """Mean per-example sum of squared errors against one-hot labels."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("class label outside probability columns")
    one_hot = np.eye(p.shape[1], dtype=float)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def precision_recall_at_k(y_true, scores, k):
    """Return precision@k and recall@k for descending anomaly scores."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s) or not len(y) or k < 1 or k > len(y):
        raise ValueError("equal non-empty arrays and 1 <= k <= n are required")
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("recall@k is undefined without positive anomalies")
    top = np.argsort(-s, kind="stable")[:k]
    found = int(y[top].sum())
    return {"precision": found / k, "recall": found / positives, "true_positives": found}
