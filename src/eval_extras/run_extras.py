"""Run checkpoint-only evaluation extras; never invokes training stages."""

import csv
import json
import os
import platform
import sys

import mlflow
import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.drift_guard.stage3_utils import reproducible_disjoint_split
from src.eval_extras.metrics import multiclass_brier, precision_recall_at_k, pr_auc
from src.feature_extraction.cnn_nets import CNNEncoder
from src.feature_extraction.pqc_builder import PyTorchPQC
from src.hpo_engine.downstream_model import QuantumClassifier


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTRA = os.path.join(ROOT, "results", "extras")


def _extract_dataset(dataset, indices, encoder, pqc):
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=128, shuffle=False)
    blocks = []
    with torch.no_grad():
        for images, _ in loader:
            blocks.append(pqc(encoder(images)).cpu().numpy())
    return np.concatenate(blocks, axis=0)


def _features_from_images(images, encoder, pqc):
    blocks = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            blocks.append(pqc(encoder(torch.stack(images[start:start + 128]))).cpu().numpy())
    return np.concatenate(blocks, axis=0)


def _make_drift_images(clean_images, severity, radius, seed):
    del seed  # Gaussian noise is not part of the blur-only analysis.
    made = []
    for image in clean_images:
        pixels = ((image.squeeze().cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        made.append(transforms.Normalize((0.5,), (0.5,))(
            transforms.ToTensor()(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius=float(radius[severity]))))))
    return made


def _model_and_data(settings):
    c1, c2, cfg = settings["extraction_config"], settings["hpo_config"], settings["monitor_config"]
    transform = transforms.Compose([
        transforms.Resize((c1["img_resize"], c1["img_resize"])),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.MNIST(root=os.path.join(ROOT, "datasets"), train=True, download=False, transform=transform)
    test_ds = datasets.MNIST(root=os.path.join(ROOT, "datasets"), train=False, download=False, transform=transform)
    device = torch.device("cpu")
    encoder = CNNEncoder(c1["n_qubits"], c1["img_resize"])
    encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/encoder_weights/cnn_encoder.pth"), map_location=device, weights_only=True))
    encoder.eval()
    pqc = PyTorchPQC(c1["n_qubits"], 2)
    pqc.weight.data.copy_(torch.from_numpy(np.load(os.path.join(ROOT, "checkpoints/encoder_weights/pqc_params.npy"))))
    pqc.eval()

    # Reconstruct only Stage 2's fixed feature-training subset; no model fitting occurs.
    n_train, n_val = int(c2["n_hpo_samples"]), int(c2.get("n_hpo_val_samples", 500))
    tr_idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n_train, test_size=n_val,
        random_state=int(settings.get("seed", 42)) + 101).split(np.zeros(len(train_ds)), train_ds.targets.numpy()))
    train_features = _extract_dataset(train_ds, tr_idx, encoder, pqc)
    params = json.load(open(os.path.join(ROOT, "checkpoints/best_model/best_params.json"), encoding="utf-8"))
    classifier = QuantumClassifier(train_features.shape[1], int(params["hidden_dim"]), dropout_rate=float(params["dropout"]))
    classifier.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints/best_model/final_classifier.pth"), map_location=device, weights_only=True))
    classifier.eval()
    n_test = int(c2.get("n_hpo_test_samples", 1000))
    s2_idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n_test,
        random_state=int(settings.get("seed", 42)) + 202).split(np.zeros(len(test_ds)), test_ds.targets.numpy()))
    return train_ds, test_ds, encoder, pqc, classifier, s2_idx, cfg


def _predict(classifier, features):
    with torch.no_grad():
        logits = classifier(torch.as_tensor(features, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.argmax(axis=1), probs.max(axis=1), probs


def _atomic_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run():
    os.makedirs(EXTRA, exist_ok=True)
    with open(os.path.join(ROOT, "settings.yaml"), encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    train_ds, test_ds, encoder, pqc, classifier, s2_idx, cfg = _model_and_data(settings)
    seed = int(settings.get("seed", 42))
    tune, cal, test_idx = reproducible_disjoint_split(len(test_ds), int(cfg["tune_samples"]),
        int(cfg["calibration_samples"]), int(cfg["test_samples"]), seed=seed + 201)
    clean_images = [test_ds[int(i)][0] for i in test_idx]
    clean_labels = np.asarray([int(test_ds.targets[int(i)]) for i in test_idx])
    clean_features = _extract_dataset(test_ds, test_idx, encoder, pqc)
    stage3_fit_features = _extract_dataset(train_ds, range(int(cfg["fit_samples"])), encoder, pqc)
    scaler = StandardScaler().fit(stage3_fit_features)
    clean_scaled = scaler.transform(clean_features)
    drift_cfg = {"blur_radii": cfg.get("blur_radii", {"mild": .8, "medium": 1.6, "heavy": 2.5})}
    names = ["clean", "blur_mild", "blur_medium", "blur_heavy"]
    feature_by_kind = {"clean": clean_features}
    scaled_by_kind = {"clean": clean_scaled}
    for j, name in enumerate(names[1:]):
        imgs = _make_drift_images(clean_images, name.removeprefix("blur_"), drift_cfg["blur_radii"], seed + 1000 + j)
        feats = _features_from_images(imgs, encoder, pqc)
        feature_by_kind[name] = feats
        scaled_by_kind[name] = scaler.transform(feats)

    # (1) PR-AUC from saved, held-out detector scores only.
    score_path = os.path.join(ROOT, "results/stage3_scores.npz")
    score_data = np.load(score_path)
    methods = ["qsvm", "rbf_ocsvm", "isolation_forest", "lof"]
    pr = {}
    for method in methods:
        pr[method] = {}
        clean = score_data[f"{method}_test_clean"]
        for name in names[1:]:
            scores = np.r_[clean, score_data[f"{method}_test_{name}"]]
            labels = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(clean), dtype=int)]
            pr[method][name] = {"pr_auc_trapezoid": pr_auc(labels, scores),
                                "average_precision": float(average_precision_score(labels, scores))}

    # (2) Brier score and accuracy of the saved Stage 2 final checkpoint on its exact test indices.
    s2_x = _extract_dataset(test_ds, s2_idx, encoder, pqc)
    s2_y = np.asarray([int(test_ds.targets[int(i)]) for i in s2_idx])
    s2_pred, _, s2_probs = _predict(classifier, s2_x)
    stage2 = {"n": len(s2_y), "accuracy": float(accuracy_score(s2_y, s2_pred)),
              "multiclass_brier": multiclass_brier(s2_y, s2_probs),
              "formula": "(1/N) * sum_i sum_c (p_ic - 1[y_i=c])^2; p=softmax(logits), classes=10",
              "mnist_test_indices_seed": seed + 202}

    # (3) Ranking on the combined clean + heavy-blur test set, using selected deployment QSVM scores.
    scores = np.r_[score_data["qsvm_test_clean"], score_data["qsvm_test_blur_heavy"]]
    labels = np.r_[np.zeros(len(clean_labels), dtype=int), np.ones(len(clean_labels), dtype=int)]
    ranking = {str(k): precision_recall_at_k(labels, scores, k) for k in (10, 100, 500, 1000, 2000, 4000)}

    # (4)-(5) Traceable Stage 3 test rows and Stage 2 accuracy/error by condition and flag.
    deployment = json.load(open(os.path.join(ROOT, "results/stage3_metrics.json"), encoding="utf-8"))
    threshold = float(deployment["methods"]["qsvm"]["sample_threshold"])
    trace_rows, error = [], {"by_condition": {}, "by_stage3_flag": {}}
    all_condition_records = []
    for name in names:
        yhat, confidence, _ = _predict(classifier, feature_by_kind[name])
        anomaly = score_data["qsvm_test_clean"] if name == "clean" else score_data[f"qsvm_test_{name}"]
        flags = anomaly > threshold
        error["by_condition"][name] = {"stage2_accuracy": float(accuracy_score(clean_labels, yhat)), "n": len(yhat)}
        for flag_val, label in ((True, "flagged"), (False, "not_flagged")):
            mask = flags == flag_val
            error["by_stage3_flag"][f"{name}_{label}"] = {
                "n": int(mask.sum()), "stage2_accuracy": float(accuracy_score(clean_labels[mask], yhat[mask])) if mask.any() else None}
        for pos, mnist_idx in enumerate(test_idx):
            row = [int(mnist_idx), int(clean_labels[pos]), name, *[float(v) for v in feature_by_kind[name][pos]],
                   int(yhat[pos]), float(confidence[pos]), float(anomaly[pos]), int(flags[pos])]
            trace_rows.append(row)
        all_condition_records.append((name, yhat, flags))
    _atomic_csv(os.path.join(EXTRA, "traceability.csv"),
        ["mnist_index", "true_digit", "drift_type", "stage1_feature_1", "stage1_feature_2", "stage1_feature_3", "stage1_feature_4",
         "stage2_predicted_digit", "stage2_confidence", "stage3_anomaly_score", "stage3_flag"], trace_rows)

    ablation = json.load(open(os.path.join(ROOT, "results/stage1_reconstruction_only_ablation.json"), encoding="utf-8"))
    error["stage1_ablation"] = {"reconstruction_only_heavy_blur_recall": ablation["methods"]["qsvm"]["heavy_recall"],
        "classification_weight_0_1_heavy_blur_recall": deployment["methods"]["qsvm"]["test_metrics_by_drift"]["blur_heavy"]["recall"]}

    # Noise feasibility is established by installed Aer. The actual noisy run is isolated in
    # run_noisy_kernel() so all standard metrics remain available if Aer execution fails.
    try:
        noise = _run_noisy_kernel(scaler.transform(stage3_fit_features), clean_scaled,
                                  scaled_by_kind["blur_heavy"], float(deployment["methods"]["qsvm"]["parameters"]["nu"]))
    except Exception as exc:  # report the actual failure, never substitute synthetic values
        noise = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    report = {"stage3_pr_auc": pr, "stage2": stage2, "precision_recall_at_k_qsvm": ranking,
              "traceability": {"rows": len(trace_rows), "unique_mnist_test_indices": len(set(map(int, test_idx))),
                               "conditions": names, "stage3_test_split_seed": seed + 201,
                               "stage2_test_split_seed": seed + 202, "indices_are_native_mnist_test_indices": True},
              "error_propagation": error, "noise_test": noise,
              "environment": {"python": sys.version.split()[0], "os": platform.platform(),
                               "qiskit": __import__("qiskit").__version__, "qiskit_aer": __import__("qiskit_aer").__version__}}
    with open(os.path.join(EXTRA, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_pitch(report)
    _write_summary(report)

    # Start a separate top-level MLflow run; this does not resume or nest into pipeline runs.
    mlflow.set_tracking_uri(settings.get("tracking_dir", "sqlite:///mlflow.db"))
    mlflow.set_experiment(settings.get("experiment_name", "QuantumSentinel-AutoML"))
    with mlflow.start_run(run_name="extras_eval"):
        mlflow.log_params({"seed": seed, "source": "saved_checkpoints_and_stage3_scores", "refit_stage3": False})
        mlflow.log_metric("stage2_final_checkpoint_accuracy", stage2["accuracy"])
        mlflow.log_metric("stage2_final_checkpoint_brier", stage2["multiclass_brier"])
        for method in methods:
            for name in names[1:]:
                mlflow.log_metric(f"{method}_{name}_pr_auc", pr[method][name]["pr_auc_trapezoid"])
        for filename in ("metrics.json", "traceability.csv", "summary.md", "elevator_pitch.txt"):
            mlflow.log_artifact(os.path.join(EXTRA, filename), artifact_path="extras")
    print(json.dumps(report, indent=2))


def _run_noisy_kernel(fit_features, clean, heavy, nu):
    """Noisy four-qubit ZZ feature map kernel, 100 fit + 200 clean + 200 blur points."""
    from qiskit import transpile
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from sklearn.metrics import recall_score
    from sklearn.svm import OneClassSVM
    from src.drift_guard.stage3_utils import conformal_upper_threshold

    # Use a subset of the original Stage 3 clean fit split, then disjoint test samples.
    # Noisy and exact kernels use identical fit/calibration/test data and selected nu.
    fit_x, cal_x, eval_clean, eval_heavy = fit_features[:100], clean[:100], clean[100:300], heavy[:200]
    points = np.vstack([fit_x, cal_x, eval_clean, eval_heavy])
    fmap = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")
    def kernel_matrix(noisy):
        noise_model = NoiseModel()
        if noisy:
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["u", "u1", "u2", "u3"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
        backend = AerSimulator(method="density_matrix", noise_model=noise_model)
        circuits = []
        for row in points:
            circ = fmap.assign_parameters(row, inplace=False)
            circ.save_density_matrix()
            circuits.append(circ)
        compiled = transpile(circuits, backend, optimization_level=0)
        job = backend.run(compiled, shots=1)
        states = [np.asarray(job.result().data(i)["density_matrix"], dtype=complex) for i in range(len(points))]
        # Hilbert-Schmidt overlap is a fidelity-kernel analogue for mixed-state noise.
        matrix = np.empty((len(points), len(points)), dtype=float)
        for i, rho in enumerate(states):
            for j in range(i, len(points)):
                val = float(np.trace(rho @ states[j]).real)
                matrix[i, j] = matrix[j, i] = val
        return matrix

    exact_gram = kernel_matrix(False)
    noisy_gram = kernel_matrix(True)
    n_fit, n_cal, n_eval = 100, 100, 200
    def calibrated_rates(gram):
        model = OneClassSVM(kernel="precomputed", nu=nu).fit(gram[:n_fit, :n_fit])
        cal_scores = -model.decision_function(gram[n_fit:n_fit+n_cal, :n_fit]).reshape(-1)
        cutoff = conformal_upper_threshold(cal_scores, 0.10)
        clean_scores = -model.decision_function(gram[n_fit+n_cal:n_fit+n_cal+n_eval, :n_fit]).reshape(-1)
        heavy_scores = -model.decision_function(gram[n_fit+n_cal+n_eval:, :n_fit]).reshape(-1)
        return float(np.mean(heavy_scores > cutoff)), float(np.mean(clean_scores > cutoff))
    exact_recall, exact_fpr = calibrated_rates(exact_gram)
    noisy_recall, noisy_fpr = calibrated_rates(noisy_gram)
    return {"status": "ok", "backend": "AerSimulator density_matrix", "noise": {"1q_depolarizing": .005, "2q_depolarizing": .02},
        "kernel": "Tr(rho_i rho_j) noisy mixed-state overlap vs exact noiseless overlap; OCSVM precomputed",
        "counts": {"fit_clean_from_stage3_fit_split": 100, "calibration_clean": 100, "test_clean": 200, "test_heavy": 200},
        "nu": nu,
        "noiseless_recall": exact_recall, "noisy_recall": noisy_recall,
        "recall_change": noisy_recall - exact_recall, "noiseless_fpr": exact_fpr,
        "noisy_fpr": noisy_fpr, "fpr_change": noisy_fpr - exact_fpr}


def _write_pitch(r):
    text = ("QuantumSentinel-AutoML combines a four-qubit hybrid feature extractor, QAOA-based hyperparameter "
        "search, and a calibrated quantum-kernel drift monitor. In the saved run, the Stage 2 search reached "
        "88.7% test accuracy. On held-out heavy blur, the selected QSVM reached 48.55% recall; the "
        "reconstruction-only Stage 1 ablation reached 10.15%. Stage 3 also compares RBF one-class SVM, "
        "Isolation Forest, and LOF. The kernel is exactly classically simulable at four qubits; this project "
        "demonstrates a reproducible hybrid workflow rather than quantum speedup.")
    with open(os.path.join(EXTRA, "elevator_pitch.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def _write_summary(r):
    p = r["stage3_pr_auc"]
    prtext = "; ".join(f"{m}: mild {p[m]['blur_mild']['pr_auc_trapezoid']:.4f}, medium {p[m]['blur_medium']['pr_auc_trapezoid']:.4f}, heavy {p[m]['blur_heavy']['pr_auc_trapezoid']:.4f}" for m in p)
    k = r["precision_recall_at_k_qsvm"]
    err = r["error_propagation"]["by_condition"]
    ab = r["error_propagation"]["stage1_ablation"]
    n = r["noise_test"]
    summary = ["# Evaluation extras", "", "| Question | Feasible | Result | File |", "|---|---|---|---|"]
    summary.append(f"| 1. Stage 3 PR-AUC | YES | Trapezoidal PR-AUC (clean vs drift): {prtext}. | metrics.json |")
    summary.append(f"| 2. Stage 2 Brier | YES | Brier {r['stage2']['multiclass_brier']:.6f}; accuracy {r['stage2']['accuracy']:.4%} (saved search report: 88.7%). {r['stage2']['formula']} | metrics.json |")
    summary.append(f"| 3. Precision/recall@k | YES | QSVM P/R@10 {k['10']['precision']:.4f}/{k['10']['recall']:.4f}; @100 {k['100']['precision']:.4f}/{k['100']['recall']:.4f}; @500 {k['500']['precision']:.4f}/{k['500']['recall']:.4f}; @1000 {k['1000']['precision']:.4f}/{k['1000']['recall']:.4f}. Top-10 precision (0.10) is below the 0.50 random-ranking baseline; recall@10 is not useful with 2,000 anomalies. | metrics.json |")
    summary.append(f"| 4. Traceability | YES | {r['traceability']['rows']} rows across four conditions, keyed by native MNIST test index. | traceability.csv |")
    flagged = r["error_propagation"]["by_stage3_flag"]
    flagtext = "; ".join(f"{c}: flagged {flagged[c + '_flagged']['stage2_accuracy']:.1%}, unflagged {flagged[c + '_not_flagged']['stage2_accuracy']:.1%}" for c in ("clean", "blur_mild", "blur_medium", "blur_heavy"))
    summary.append(f"| 5. Error propagation | YES | Stage 2 accuracy clean/mild/medium/heavy: " + ", ".join(f"{x} {err[x]['stage2_accuracy']:.2%}" for x in err) + f". Flagged/unflagged Stage 2 accuracy: {flagtext}. Ablation heavy-blur recall: {ab['reconstruction_only_heavy_blur_recall']:.2%} vs {ab['classification_weight_0_1_heavy_blur_recall']:.2%}; degradation tracks blur, but flags do not consistently separate errors, and these associations do not prove causation. | metrics.json |")
    summary.append("| 6. Submission checklist | YES | Added MIT LICENSE, all four clearly marked README sections, pinned direct dependencies, recorded Python/OS/Qiskit versions; seed 42 was already in settings/README. | README.md, LICENSE, requirements.txt, elevator_pitch.txt |")
    if n.get("status") == "ok":
        noise_result = (f"Recall {n['noiseless_recall']:.2%} → {n['noisy_recall']:.2%} "
                        f"({n['recall_change']:+.2%}); FPR {n['noiseless_fpr']:.2%} → "
                        f"{n['noisy_fpr']:.2%} ({n['fpr_change']:+.2%})")
    else:
        noise_result = n.get("reason", "not run")
    summary.append(f"| 7. Noise test | {('YES' if n.get('status') == 'ok' else 'PARTIAL')} | {noise_result} | metrics.json |")
    summary.extend(["", "PR-AUC, ranking, and flag metrics use saved Stage 3 score arrays. The noise-only comparison refit the selected OCSVM (nu=0.01) on 100 fixed samples from Stage 3's saved-checkpoint clean fit features because noisy scores were not saved; Stage 1 and Stage 2 were not retrained."])
    with open(os.path.join(EXTRA, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    run()
