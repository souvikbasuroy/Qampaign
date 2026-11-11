"""Nested train-only response modeling with calibrated and quantum-kernel comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.eval_extras.metrics import pr_auc


class CampaignFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic campaign transforms; no statistics are learned from held-out rows."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        balance = pd.to_numeric(frame["balance"], errors="coerce").fillna(0.0)
        campaign = pd.to_numeric(frame["campaign"], errors="coerce").fillna(0.0)
        pdays = pd.to_numeric(frame["pdays"], errors="coerce").fillna(-1.0)
        previous = pd.to_numeric(frame["previous"], errors="coerce").fillna(0.0)
        outcome = frame["poutcome"].astype(str).str.strip().str.lower()
        frame["balance_log_signed"] = np.sign(balance) * np.log1p(np.abs(balance))
        frame["campaign_log"] = np.log1p(np.maximum(campaign, 0.0))
        frame["previously_contacted"] = (pdays >= 0).astype(int)
        frame["previous_positive"] = (previous > 0).astype(int)
        frame["poutcome_success"] = (outcome == "success").astype(int)
        return frame

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features) + [
            "balance_log_signed", "campaign_log", "previously_contacted",
            "previous_positive", "poutcome_success",
        ], dtype=object)


class FeatureMapScale(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float) * float(self.scale), 0.0, np.pi)


class FastZZKernel:
    """Cached exact-statevector ZZ fidelity kernel."""
    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.cache = {}

    def _states(self, X):
        states = []
        for row in np.asarray(X, dtype=float):
            key = tuple(np.round(row, 9))
            if key not in self.cache:
                circuit = self.feature_map.assign_parameters(row)
                self.cache[key] = Statevector.from_instruction(circuit).data
            states.append(self.cache[key])
        return np.asarray(states)

    def evaluate(self, X, Y=None):
        left = self._states(X)
        right = left if Y is None else self._states(Y)
        return np.abs(left @ right.conj().T) ** 2


class QuantumKernelSVC(ClassifierMixin, BaseEstimator):
    """SVC with a ZZ fidelity kernel; ranking uses decision_function, never SVC probability."""
    def __init__(self, C=1.0, reps=1):
        self.C = C
        self.reps = reps

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.qubit_count_ = int(X.shape[1])
        if self.qubit_count_ not in (4, 6):
            raise ValueError("This experiment configures four or six qubits.")
        self.feature_map_ = ZZFeatureMap(
            feature_dimension=self.qubit_count_, reps=int(self.reps), entanglement="linear"
        )
        self.kernel_ = FastZZKernel(self.feature_map_)
        self.X_fit_ = X.copy()
        self.svc_ = SVC(C=float(self.C), kernel="precomputed", class_weight="balanced")
        self.svc_.fit(self.kernel_.evaluate(X), y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(
            self.kernel_.evaluate(np.asarray(X, dtype=float), self.X_fit_)
        )

    def predict(self, X):
        return self.svc_.predict(self.kernel_.evaluate(np.asarray(X, dtype=float), self.X_fit_))


def _feature_columns(cfg, mode):
    definition = cfg["feature_sets"][mode]
    return list(definition["numeric"]), list(definition["categorical"])


def _preprocessor(cfg, mode):
    numeric, categorical = _feature_columns(cfg, mode)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop", verbose_feature_names_out=True,
    )


def _logistic_pipeline(cfg, mode):
    return Pipeline([
        ("engineer", CampaignFeatureEngineer()),
        ("preprocess", _preprocessor(cfg, mode)),
        ("classifier", LogisticRegression(
            max_iter=3000, solver="lbfgs", random_state=int(cfg.get("seed", 42)),
        )),
    ])


def _quantum_pipeline(cfg, qubits, reps, c_value, scale):
    return Pipeline([
        ("engineer", CampaignFeatureEngineer()),
        ("preprocess", _preprocessor(cfg, "compact")),
        ("reduce", TruncatedSVD(n_components=int(qubits), random_state=int(cfg.get("seed", 42)))),
        ("angle_range", MinMaxScaler(feature_range=(0.0, np.pi), clip=True)),
        ("angle_scale", FeatureMapScale(scale=float(scale))),
        ("classifier", QuantumKernelSVC(C=float(c_value), reps=int(reps))),
    ])


def candidate_grid(cfg):
    """Return the equal 40-setting model-family budgets for audit/logging."""
    weights = [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 4}, {0: 1, 1: 6}]
    classical = [
        {"family": "logistic", "feature_mode": mode, "C": float(c), "class_weight": weight}
        for mode in ("all", "compact") for c in cfg["logistic_c"] for weight in weights
    ]
    quantum = [
        {"family": "quantum_zz_svc", "qubit_count": int(q), "reps": int(reps),
         "C": float(c), "kernel_scale": float(scale), "feature_map": "ZZFeatureMap",
         "angle_range": "[0, pi]", "backend": "statevector", "shots": 0}
        for q in cfg["quantum_qubit_counts"] for reps in cfg["quantum_reps"]
        for c in cfg["quantum_c"] for scale in cfg["quantum_kernel_scales"]
    ]
    return classical + quantum


def _inner_splits(X, y, cfg, seed):
    splitter = StratifiedKFold(
        n_splits=int(cfg["inner_cv_folds"]), shuffle=True, random_state=int(seed)
    )
    return list(splitter.split(X, y))


def _tune_logistic(X, y, cfg, splits):
    weights = [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 4}, {0: 1, 1: 6}]
    options = []
    for mode in ("all", "compact"):
        grid = GridSearchCV(
            _logistic_pipeline(cfg, mode),
            {"classifier__C": [float(c) for c in cfg["logistic_c"]],
             "classifier__class_weight": weights},
            scoring="neg_log_loss", cv=splits, refit=True, n_jobs=1,
        )
        grid.fit(X, y)
        options.append((float(grid.best_score_), mode, grid))
    score, mode, grid = max(options, key=lambda item: (item[0], item[1] == "compact"))
    return grid.best_estimator_, {"feature_mode": mode, **grid.best_params_,
                                 "inner_neg_log_loss": score,
                                 "settings_evaluated": 40}


def _tune_quantum(X, y, cfg, splits):
    """Tune 40 settings while reusing each inner-fold feature map kernel across C."""
    records = []
    for qubits in cfg["quantum_qubit_counts"]:
        for fit_idx, valid_idx in splits:
            X_fit, X_valid = X.iloc[fit_idx], X.iloc[valid_idx]
            y_fit, y_valid = y[fit_idx], y[valid_idx]
            engineer = CampaignFeatureEngineer().fit(X_fit)
            train_engineered = engineer.transform(X_fit)
            valid_engineered = engineer.transform(X_valid)
            prep = _preprocessor(cfg, "compact").fit(train_engineered, y_fit)
            train_encoded = prep.transform(train_engineered)
            valid_encoded = prep.transform(valid_engineered)
            reduce = TruncatedSVD(n_components=int(qubits), random_state=int(cfg.get("seed", 42)))
            train_reduced = reduce.fit_transform(train_encoded)
            valid_reduced = reduce.transform(valid_encoded)
            angle = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
            train_angles = angle.fit_transform(train_reduced)
            valid_angles = angle.transform(valid_reduced)
            for reps in cfg["quantum_reps"]:
                for scale in cfg["quantum_kernel_scales"]:
                    train_x = np.clip(train_angles * float(scale), 0.0, np.pi)
                    valid_x = np.clip(valid_angles * float(scale), 0.0, np.pi)
                    fmap = ZZFeatureMap(feature_dimension=int(qubits), reps=int(reps), entanglement="linear")
                    kernel = FastZZKernel(fmap)
                    train_kernel = kernel.evaluate(train_x)
                    valid_kernel = kernel.evaluate(valid_x, train_x)
                    for c_value in cfg["quantum_c"]:
                        estimator = SVC(C=float(c_value), kernel="precomputed", class_weight="balanced")
                        estimator.fit(train_kernel, y_fit)
                        scores = estimator.decision_function(valid_kernel)
                        records.append({
                            "qubit_count": int(qubits), "reps": int(reps),
                            "kernel_scale": float(scale), "C": float(c_value),
                            "fold_auc": float(roc_auc_score(y_valid, scores)),
                        })
    grouped = {}
    for row in records:
        key = (row["qubit_count"], row["reps"], row["kernel_scale"], row["C"])
        grouped.setdefault(key, []).append(row["fold_auc"])
    key, scores = max(grouped.items(), key=lambda item: (float(np.mean(item[1])), item[0]))
    qubits, reps, scale, c_value = key
    chosen = _quantum_pipeline(cfg, qubits, reps, c_value, scale)
    chosen.fit(X, y)
    return chosen, {
        "qubit_count": int(qubits), "reps": int(reps), "kernel_scale": float(scale),
        "C": float(c_value), "inner_roc_auc": float(np.mean(scores)),
        "feature_mode": "compact", "feature_map": "ZZFeatureMap",
        "settings_evaluated": 40,
    }


def _pooled_metrics(y, probabilities, raw_scores=None):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    positives = max(int(y.sum()), 1)
    result = {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(pr_auc(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
    }
    for k in (10, 20, 30):
        selected = np.argsort(-p, kind="stable")[:min(k, len(y))]
        result[f"precision_at_{k}"] = float(y[selected].mean())
        result[f"recall_at_{k}"] = float(y[selected].sum() / positives)
    if raw_scores is not None:
        raw = np.asarray(raw_scores, dtype=float)
        result["decision_roc_auc"] = float(roc_auc_score(y, raw))
        result["decision_pr_auc"] = float(pr_auc(y, raw))
        for k in (10, 20, 30):
            selected_raw = np.argsort(-raw, kind="stable")[:min(k, len(y))]
            result[f"decision_precision_at_{k}"] = float(y[selected_raw].mean())
            result[f"decision_recall_at_{k}"] = float(y[selected_raw].sum() / positives)
    return result


def _bootstrap_metric_intervals(y, probabilities, seed, n_bootstrap=1000):
    """Stratified bootstrap of OOF (label, probability) pairs; preserves the observed rate."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(int(seed))
    samples = {}
    for _ in range(int(n_bootstrap)):
        indices = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        rng.shuffle(indices)
        try:
            metrics = _pooled_metrics(y[indices], p[indices])
        except ValueError:
            continue
        for name, value in metrics.items():
            samples.setdefault(name, []).append(float(value))
    return {name: {"ci95_low": float(np.quantile(values, .025)),
                   "ci95_high": float(np.quantile(values, .975)),
                   "bootstrap_replicates": len(values), "method": "stratified row bootstrap of pooled OOF pairs"}
            for name, values in samples.items()}


def _summarize_repeats(repeat_metrics):
    summaries = {}
    for name in repeat_metrics[0]:
        values = np.asarray([row[name] for row in repeat_metrics], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half = float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summaries[name] = {"mean": mean, "std_across_repeats": std,
                           "ci95_low": mean - half, "ci95_high": mean + half,
                           "repeat_values": values.tolist()}
    return summaries


def _calibration_table(y, p, bins=8):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    boundaries = np.quantile(p, np.linspace(0, 1, int(bins) + 1))
    groups = []
    for index in range(int(bins)):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (p >= lower) & (p <= upper if index == bins - 1 else p < upper)
        if np.any(mask):
            groups.append((f"quantile_{index + 1}", np.flatnonzero(mask)))
    # Explicit upper-tail bin so viewers can see extrapolation beyond the calibrated OOF range.
    threshold = float(np.quantile(p, 0.9))
    tail = np.flatnonzero(p >= threshold)
    if len(tail):
        groups.append(("top_10_percent", tail))
    rows = []
    z = 1.959963984540054
    for name, group in groups:
        n = len(group)
        observed = float(y[group].mean())
        denom = 1 + z * z / n
        center = (observed + z * z / (2 * n)) / denom
        half = z * np.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / denom
        rows.append({"bin": name, "n": n, "mean_predicted": float(p[group].mean()),
                     "max_predicted": float(p[group].max()), "observed_rate": observed,
                     "ci95_low": max(0.0, center - half), "ci95_high": min(1.0, center + half)})
    return rows


def _kernel_alignment(K, y):
    K = np.asarray(K, dtype=float)
    target = np.asarray(y, dtype=float) * 2 - 1
    target = np.outer(target, target)
    center = np.eye(len(K)) - np.ones_like(K) / len(K)
    kc = center @ K @ center
    tc = center @ target @ center
    denom = np.linalg.norm(kc) * np.linalg.norm(tc)
    return float(np.sum(kc * tc) / denom) if denom else 0.0


def _write_artifacts(out, nested_results, calibration, kernel_matrix, alignment, feature_map_order):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, result in nested_results.items():
        for metric, stats in result["summary"].items():
            bootstrap = result.get("bootstrap_ci95", {}).get(metric, {})
            rows.append({"model": family, "metric": metric, "mean": stats["mean"],
                         "std_across_repeats": stats["std_across_repeats"],
                         "repeated_cv_ci95_low": stats["ci95_low"],
                         "repeated_cv_ci95_high": stats["ci95_high"],
                         "bootstrap_ci95_low": bootstrap.get("ci95_low"),
                         "bootstrap_ci95_high": bootstrap.get("ci95_high")})
    pd.DataFrame(rows).to_csv(out / "stage1_cv_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(out / "stage1_calibration_curve.csv", index=False)
    pd.DataFrame(feature_map_order).to_csv(out / "stage1_quantum_qubit_feature_order.csv", index=False)
    pd.DataFrame(kernel_matrix).to_csv(out / "stage1_quantum_kernel_matrix.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    models = list(nested_results)
    for metric, marker in (("roc_auc", "o"), ("pr_auc", "s"), ("log_loss", "^") ):
        means = [nested_results[name]["summary"][metric]["mean"] for name in models]
        lows = [means[i] - nested_results[name]["summary"][metric]["ci95_low"] for i, name in enumerate(models)]
        highs = [nested_results[name]["summary"][metric]["ci95_high"] - means[i] for i, name in enumerate(models)]
        ax.errorbar(models, means, yerr=[lows, highs], marker=marker, capsize=3, label=metric)
    ax.set_title("Nested repeated CV; error bars are 95% CIs")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "stage1_cv_comparison.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot([row["mean_predicted"] for row in calibration],
            [row["observed_rate"] for row in calibration], marker="o")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed response rate",
           title="Nested out-of-fold calibration; top decile included")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "stage1_calibration.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(kernel_matrix, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="ZZ fidelity")
    ax.set(title=f"Train-only ZZ kernel; target alignment={alignment:.3f}",
           xlabel="Training sample", ylabel="Training sample")
    fig.tight_layout(); fig.savefig(out / "stage1_quantum_kernel_matrix.png", dpi=170); plt.close(fig)


def rank_predictions(predictions):
    return predictions.sort_values(["predicted_probability", "customer_id"],
                                   ascending=[False, True], kind="mergesort").reset_index(drop=True)


def choose_cv_candidate(cv_results):
    return min(cv_results, key=lambda key: (cv_results[key]["summary"]["log_loss"]["mean"], key))


def validate_stage1_inputs(train, test):
    feature_names = set(train.columns) | set(test.columns)
    if "y" in test.columns or "duration" in feature_names:
        raise ValueError("Target leakage guard: test labels or duration are not allowed.")


def _fit_calibrator(base, X, y, cfg, seed):
    cv = StratifiedKFold(n_splits=int(cfg["calibration_folds"]), shuffle=True, random_state=int(seed))
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)
    calibrated.fit(X, y)
    return calibrated


def _qubit_order(best_quantum, X_train, y_train):
    best_quantum.fit(X_train, y_train)
    prep = best_quantum.named_steps["preprocess"]
    reduced = best_quantum.named_steps["reduce"]
    source_names = prep.get_feature_names_out()
    order = []
    for qubit, component in enumerate(reduced.components_):
        top = np.argsort(-np.abs(component))[:4]
        order.append({"qubit": qubit, "component": f"SVD_PC{qubit + 1}",
                      "top_source_features": ", ".join(str(source_names[i]) for i in top),
                      "angle_range": "[0, pi]"})
    kernel_estimator = best_quantum.named_steps["classifier"]
    X_work = best_quantum.named_steps["engineer"].transform(X_train)
    X_work = best_quantum.named_steps["preprocess"].transform(X_work)
    X_work = best_quantum.named_steps["reduce"].transform(X_work)
    X_work = best_quantum.named_steps["angle_range"].transform(X_work)
    X_angle = best_quantum.named_steps["angle_scale"].transform(X_work)
    matrix = kernel_estimator.kernel_.evaluate(X_angle)
    alignment = _kernel_alignment(matrix, y_train)
    return order, matrix, alignment


def run_stage1(train, test, cfg, seed, out_dir):
    validate_stage1_inputs(train, test)
    y = train["y"].astype(int).to_numpy()
    train_X = train.drop(columns=["customer_id", "y"])
    test_X = test.drop(columns=["customer_id", "test_row"], errors="ignore")
    folds, repeats = int(cfg["cv_folds"]), int(cfg["cv_repeats"])
    shared_splits = list(RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=int(seed)
    ).split(train_X, y))
    repeat_y = [y.copy() for _ in range(repeats)]
    outputs = {
        "nested_logistic": {"prob": [np.full(len(y), np.nan) for _ in range(repeats)]},
        "nested_quantum": {"prob": [np.full(len(y), np.nan) for _ in range(repeats)],
                           "decision": [np.full(len(y), np.nan) for _ in range(repeats)]},
    }
    fold_selections = []
    quantum_fold_auc = []
    for split_index, (fit_idx, valid_idx) in enumerate(shared_splits):
        repeat_id, fold_id = divmod(split_index, folds)
        X_fit, X_valid = train_X.iloc[fit_idx], train_X.iloc[valid_idx]
        y_fit, y_valid = y[fit_idx], y[valid_idx]
        inner = _inner_splits(X_fit, y_fit, cfg, seed + split_index)

        logistic, logistic_config = _tune_logistic(X_fit, y_fit, cfg, inner)
        logistic_decision = logistic.decision_function(X_valid)
        logistic_calibrated = _fit_calibrator(logistic, X_fit, y_fit, cfg, seed + 2000 + split_index)
        logistic_prob = logistic_calibrated.predict_proba(X_valid)[:, 1]
        outputs["nested_logistic"]["prob"][repeat_id][valid_idx] = logistic_prob

        quantum, quantum_config = _tune_quantum(X_fit, y_fit, cfg, inner)
        quantum_decision = quantum.decision_function(X_valid)
        quantum_calibrated = _fit_calibrator(quantum, X_fit, y_fit, cfg, seed + 4000 + split_index)
        quantum_prob = quantum_calibrated.predict_proba(X_valid)[:, 1]
        outputs["nested_quantum"]["prob"][repeat_id][valid_idx] = quantum_prob
        outputs["nested_quantum"]["decision"][repeat_id][valid_idx] = quantum_decision
        quantum_fold_auc.append({
            "repeat": repeat_id + 1, "fold": fold_id + 1,
            "decision_roc_auc": float(roc_auc_score(y_valid, quantum_decision)),
            "probability_roc_auc": float(roc_auc_score(y_valid, quantum_prob)),
        })
        fold_selections.append({"repeat": repeat_id + 1, "fold": fold_id + 1,
                                "logistic": logistic_config, "quantum": quantum_config})

    nested_results = {}
    for family, data in outputs.items():
        repeats_metrics = []
        for repeat in range(repeats):
            prob = data["prob"][repeat]
            if np.isnan(prob).any():
                raise AssertionError("Nested outer CV failed to predict every training row.")
            raw = data.get("decision", [None] * repeats)[repeat] if "decision" in data else None
            repeats_metrics.append(_pooled_metrics(y, prob, raw_scores=raw))
        pooled_mean = np.mean(data["prob"], axis=0)
        nested_results[family] = {
            "summary": _summarize_repeats(repeats_metrics),
            "bootstrap_ci95": _bootstrap_metric_intervals(y, pooled_mean, seed + (0 if family == "nested_logistic" else 100), 1000),
            "repeat_metrics": repeats_metrics,
            "pooled_oof_probability_mean": pooled_mean.tolist(),
            "top_k_by_repeat": [
                {"k": k, "customer_ids": train.iloc[np.argsort(-data["prob"][repeat], kind="stable")[:k]]["customer_id"].astype(str).tolist(),
                 "positives_found": int(y[np.argsort(-data["prob"][repeat], kind="stable")[:k]].sum()),
                 "poutcome_success_count": int((train.iloc[np.argsort(-data["prob"][repeat], kind="stable")[:k]]["poutcome"].astype(str).str.lower().to_numpy() == "success").sum())}
                for repeat in range(repeats) for k in (10, 20, 30)
            ],
        }
    by_repeat_auc = []
    for repeat in range(1, repeats + 1):
        fold_rows = [row for row in quantum_fold_auc if row["repeat"] == repeat]
        by_repeat_auc.append({
            "decision_roc_auc": float(np.mean([row["decision_roc_auc"] for row in fold_rows])),
            "probability_roc_auc": float(np.mean([row["probability_roc_auc"] for row in fold_rows])),
        })
    nested_results["nested_quantum"]["outer_fold_auc_comparison"] = {
        "method": "mean fold ROC-AUC within each outer repeat, then mean/95% CI across repeats",
        "decision_function": _summarize_repeats([{"roc_auc": row["decision_roc_auc"]} for row in by_repeat_auc])["roc_auc"],
        "platt_predict_proba": _summarize_repeats([{"roc_auc": row["probability_roc_auc"]} for row in by_repeat_auc])["roc_auc"],
        "fold_values": quantum_fold_auc,
    }
    chosen_family = min(nested_results,
                        key=lambda name: (nested_results[name]["summary"]["log_loss"]["mean"], name))

    full_inner = _inner_splits(train_X, y, cfg, seed + 9000)
    final_logistic, final_logistic_config = _tune_logistic(train_X, y, cfg, full_inner)
    final_quantum, final_quantum_config = _tune_quantum(train_X, y, cfg, full_inner)
    final_base = final_logistic if chosen_family == "nested_logistic" else final_quantum
    final_calibrated = _fit_calibrator(final_base, train_X, y, cfg, seed + 12000)
    probabilities = final_calibrated.predict_proba(test_X)[:, 1]
    predictions = rank_predictions(pd.DataFrame({
        "customer_id": test["customer_id"].astype(str).to_numpy(),
        "predicted_probability": probabilities,
    }))

    # Diagnostic fit is train-only even when nested CV selected the classical model.
    qubit_order, kernel_matrix, alignment = _qubit_order(final_quantum, train_X, y)
    calibration_oof = np.asarray(nested_results[chosen_family]["pooled_oof_probability_mean"], dtype=float)
    calibration = _calibration_table(y, calibration_oof, cfg["calibration_bins"])
    calibration_range = [float(calibration_oof.min()), float(calibration_oof.max())]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "stage1_predictions.csv", index=False)
    _write_artifacts(out, nested_results, calibration, kernel_matrix, alignment, qubit_order)

    result = {
        "train_rows": len(train), "test_rows": len(test), "positive_train": int(y.sum()),
        "base_rate": float(y.mean()), "seed": int(seed),
        "validation": "Nested repeated stratified CV: outer 5x5; inner 3-fold model/feature selection; train only",
        "calibration": "Platt/sigmoid on outer-training folds; no isotonic; final test prediction made once",
        "tuning_budget_per_family": 40,
        "same_outer_and_inner_folds_for_both_families": True,
        "chosen_family_by_nested_cv_log_loss": chosen_family,
        "final_logistic_config": final_logistic_config,
        "final_quantum_config": final_quantum_config,
        "quantum_metadata": {
            **final_quantum_config, "backend": "exact statevector fidelity kernel",
            "shots": 0, "seed": int(seed),
            "ranking_score": "SVC decision_function; probability=True is not used",
            "angle_range": "MinMaxScaler fitted on training fold to [0, pi], then kernel scale in {0.5, 1.0}",
            "qubit_feature_order": qubit_order,
            "kernel_target_alignment_train_only": alignment,
        },
        "candidate_results": nested_results,
        "precision_recall_k_values": [10, 20, 30],
        "fold_selections": fold_selections,
        "top_group_diagnostic": {
            "train_poutcome_success_count": int((train["poutcome"].astype(str).str.lower() == "success").sum()),
            "train_poutcome_success_positive_count": int(((train["poutcome"].astype(str).str.lower() == "success") & (y == 1)).sum()),
            "same_top10_customer_ids_each_repeat": len({tuple(sorted(row["customer_ids"])) for row in nested_results[chosen_family]["top_k_by_repeat"] if row["k"] == 10}) == 1,
            "top10_repeat_detail": [row for row in nested_results[chosen_family]["top_k_by_repeat"] if row["k"] == 10],
            "interpretation": "If repeat top-ten IDs are identical and are the poutcome=success group, Precision@10 is dominated by that single feature group.",
        },
        "calibration_curve": calibration,
        "calibration_oof_probability_range": calibration_range,
        "test_top10_range": [float(predictions.head(10).predicted_probability.min()),
                             float(predictions.head(10).predicted_probability.max())],
        "test_top10_above_calibrated_oof_range": bool(
            float(predictions.head(10).predicted_probability.min()) > calibration_range[1]
        ),
        "test_predictions_file": str(out / "stage1_predictions.csv"),
        "top10": predictions.head(10).to_dict(orient="records"),
    }
    (out / "stage1_cv_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Quantum feature order:", json.dumps(qubit_order))
    print("Train-only quantum kernel target alignment:", alignment)
    return result, predictions
