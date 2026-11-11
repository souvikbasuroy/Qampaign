"""Workbook-only three-stage customer prediction, selection, and simulation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import mlflow
import numpy as np
import yaml
from qiskit_algorithms.utils import algorithm_globals

from src.customer_pipeline.stage1 import run_stage1
from src.customer_pipeline.stage2 import solve_selection
from src.customer_pipeline.stage3 import run_simulation
from src.customer_pipeline.workbook import load_participant_workbook
from src.customer_pipeline.submission_artifacts import export_submission_artifacts


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    algorithm_globals.random_seed = int(seed)


def _announce_mapping(stage, note):
    print(f"ID_MAPPING_NOTE [{stage}]: {note}")


def _log_cv_metrics(stage1):
    for candidate, result in stage1["candidate_results"].items():
        for metric, summary in result["summary"].items():
            for suffix in ("mean", "std_across_repeats", "ci95_low", "ci95_high"):
                mlflow.log_metric(
                    f"stage1_{candidate}_{metric}_{suffix}",
                    float(summary[suffix]),
                )


def run_pipeline(settings_path="settings.yaml"):
    with open(settings_path, encoding="utf-8-sig") as handle:
        settings = yaml.safe_load(handle)
    cfg = settings["customer_project"]
    seed = int(cfg["seed"])
    _seed_everything(seed)

    workbook = load_participant_workbook(
        cfg["workbook_path"], cfg.get("id_mapping")
    )
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_status, mapping_note = workbook["mapping_status"], workbook["mapping_note"]

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run() as run:
        mlflow.log_params({
            "seed": seed,
            "workbook": cfg["workbook_path"],
            "id_mapping_setting": str(cfg.get("id_mapping")),
            "id_mapping_status": mapping_status,
            "stage1_train_rows": len(workbook["train"]),
            "stage1_test_rows": len(workbook["test"]),
            "stage1_train_base_rate": workbook["train_base_rate"],
            "stage1_cv": f'nested outer {cfg["stage1"]["cv_repeats"]}x{cfg["stage1"]["cv_folds"]}; inner {cfg["stage1"]["inner_cv_folds"]}-fold train-only search',
            "stage1_calibration": "Platt sigmoid in outer training data; no isotonic",
            "stage1_candidate_configs_per_family": 40,
            "stage1_bootstrap_replicates": 1000,
            "stage2_qaoa_shots_each_of_80_settings": cfg["stage2"]["qaoa_shots"],
            "stage2_qaoa_seed": seed,
            "stage1_selected_count": workbook["constraints"]["K"],
            "stage2_candidate_count": workbook["constraints"]["N"],
            "stage2_budget_limit_from_workbook": workbook["constraints"]["C"],
            "stage2_capacity_limit_from_workbook": workbook["constraints"]["H"],
            "stage3_seed": seed,
            "stage3_shots": cfg["stage3"]["shots"],
            "stage3_backend": "Qiskit AerSimulator",
        })

        _announce_mapping("Stage 1", mapping_note)
        stage1, predictions = run_stage1(
            workbook["train"], workbook["test"], cfg["stage1"], seed, out_dir
        )
        print(f"Stage 1 selected {stage1['chosen_family_by_nested_cv_log_loss']} using nested-CV log loss; base rate={stage1['base_rate']:.3%}; top 10 IDs={stage1['top10']}")
        stage1["id_mapping_status"] = mapping_status
        stage1["id_mapping_warning"] = mapping_note
        (out_dir / "stage1_cv_metrics.json").write_text(
            json.dumps(stage1, indent=2), encoding="utf-8"
        )
        _log_cv_metrics(stage1)

        top_n = int(workbook["constraints"]["N"])
        top10 = predictions.head(top_n).copy()
        _announce_mapping("Stage 2", mapping_note)
        selected, stage2 = solve_selection(
            top10, workbook["stage2_data"], workbook["constraints"],
            cfg["stage2"], seed, out_dir, mapping_status, mapping_note,
            robustness_candidate_pool=predictions.head(top_n + 1).copy(),
        )
        print(f"Stage 2 selected IDs={stage2['exact']['selected_ids']}; objective={stage2['exact']['objective_value']:.6f}; cost={stage2['exact']['contact_cost_used']}/{stage2['exact']['contact_cost_limit']}; capacity={stage2['exact']['capacity_used']}/{stage2['exact']['capacity_limit']}")
        mlflow.log_metric("stage2_exact_objective_value", stage2["exact"]["objective_value"])
        mlflow.log_metric("stage2_feasible_subsets", stage2["exact"]["feasible_combinations"])
        for metric_name, metric_value in {
            "stage2_qaoa_approximation_ratio": stage2["qaoa"]["approximation_ratio_to_exact"],
            "stage2_qaoa_feasibility_rate": stage2["qaoa"]["feasibility_rate"],
            "stage2_qaoa_exact_optimum_probability_mass": stage2["qaoa"]["exact_optimum_probability_mass"],
            "stage2_qaoa_best_sampled_objective": stage2["qaoa"]["best_sampled_objective"],
            "stage2_qaoa_median_optimum_mass_80_runs": stage2["qaoa"]["best_of_80_measured_runs"]["median_optimum_probability_mass"],
            "stage2_qaoa_runs_ge_10pct_optimum_mass": stage2["qaoa"]["best_of_80_measured_runs"]["runs_at_or_above_threshold"],
        }.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, float(metric_value))

        _announce_mapping("Stage 3", mapping_note)
        stage3 = run_simulation(
            selected, cfg["stage3"], seed, out_dir, mapping_status, mapping_note
        )
        mlflow.log_metric(
            "stage3_estimated_success_probability",
            stage3["estimated_success_probability"],
        )
        mlflow.log_metric("stage3_exact_success_probability", stage3["exact_probability_from_same_six_probabilities"])
        for row in stage3["sensitivity_table"]:
            mlflow.log_metric(
                "stage3_sensitivity_" + row["scenario"],
                float(row["probability_at_least_3_of_6"]),
            )
        mlflow.log_metric("stage3_success_ci95_low", stage3["estimated_success_ci95"][0])
        mlflow.log_metric("stage3_success_ci95_high", stage3["estimated_success_ci95"][1])

        # Supplemental judge-facing outputs are derived from the completed stage artifacts.
        package_root = Path(__file__).resolve().parents[2]
        configured_submission_dir = Path(cfg.get("submission_output_dir", ".."))
        if not configured_submission_dir.is_absolute():
            configured_submission_dir = (Path(settings_path).resolve().parent / configured_submission_dir).resolve()
        diagnostics_cfg = cfg.get("submission_diagnostics", {})
        export_submission_artifacts(package_root=package_root, repository_root=configured_submission_dir,
            noise_shots=int(diagnostics_cfg.get("noise_shots", 20000)),
            mc_draws=int(diagnostics_cfg.get("monte_carlo_draws", 1000000)),
            uncertainty_draws=int(diagnostics_cfg.get("uncertainty_draws", 10000)),
            optimization_level=int(diagnostics_cfg.get("transpiler_optimization_level", 1)),
            seed_transpiler=int(diagnostics_cfg.get("seed_transpiler", seed)))

        summary = {
            "run_id": run.info.run_id,
            "seed": seed,
            "workbook": cfg["workbook_path"],
            "id_mapping_status": mapping_status,
            "id_mapping_warning": mapping_note,
            "stage1": {
                "chosen_family_by_nested_cv_log_loss": stage1["chosen_family_by_nested_cv_log_loss"],
                "base_rate": stage1["base_rate"],
                "top10_ids": stage1["top10"],
                "cv_file": "stage1_cv_metrics.json",
                "prediction_file": "stage1_predictions.csv",
            },
            "stage2": {
                "selected_ids": stage2["exact"]["selected_ids"],
                "objective_value": stage2["exact"]["objective_value"],
                "contact_cost": f'{stage2["exact"]["contact_cost_used"]}/{stage2["exact"]["contact_cost_limit"]}',
                "capacity": f'{stage2["exact"]["capacity_used"]}/{stage2["exact"]["capacity_limit"]}',
                "qaoa_best_feasible": stage2["qaoa"]["best_feasible_sample"],
                "qaoa_best_of_80": stage2["qaoa"]["best_of_80_measured_runs"],
                "top_ten_pool_sensitivity": stage2["top_ten_pool_sensitivity"],
            },
            "stage3": {
                "estimated_probability": stage3["estimated_success_probability"],
                "ci95": stage3["estimated_success_ci95"],
                "exact_probability": stage3["exact_probability_from_same_six_probabilities"],
                "sensitivity": stage3["sensitivity_table"],
            },
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(out_dir), artifact_path="workbook_pipeline")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_pipeline()
