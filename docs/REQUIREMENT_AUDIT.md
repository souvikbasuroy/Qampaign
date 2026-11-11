# Atomiq Fall Fest 2026 requirements audit

Audit scope: repository root and the standalone Qampaign package. A1-D publication status can only be completed by the project owner; execution checks are recorded after the package environment checks.

| Requirement | Status | Evidence | Gap | Planned fix |
|---|---|---|---|---|
| A1 Public GitHub repository and runnable notebook/scripts | Human task | notebooks/customer_selection_workbook.ipynb; main.py; src/customer_pipeline/; README.md | Code is present; public visibility cannot be verified or changed locally. | Owner publishes the repository; notebook and script execution verified locally. |
| A2 Pinned requirements | Met | requirements.txt; README.md, “Setup and reproducibility” | None. | None. |
| A3 Seeds for splitting, training, optimizer, transpiler, simulator | Met | results/RUN_INFO.md, “Reproducibility”; settings.yaml | None. | None. |
| A4 Open-source license | Met | LICENSE, MIT text | None. | None. |
| B1 Root Stage 1 CSV | Met | stage1_predictions.csv; columns customer_id, predicted_probability, rank, is_top_10 | None. | None. |
| B2 Root Stage 2 CSV/JSON | Met | stage2_selection.csv; stage2_selection.json, keys top_10_candidate_ids, selected_six_ids, budget/capacity used and limits, objective, measured bitstrings/counts, brute-force optimum, QAOA and submitted optimality gaps | None. | None. |
| B3 Root Stage 3 CSV/JSON | Met | stage3_simulation.csv; stage3_simulation.json, qubit_order, circuit_parameters, counts, quantum estimate, exact result, Monte Carlo, errors, transpilation, noise and uncertainty propagation | None. | None. |
| C1 Methodology and formulation | Met | README.md, “Pipeline overview and methodology” | None. | None. |
| C2 Classical/quantum separation and claims | Met | README.md, “Classical vs quantum rigor”; separate Stage 1 metrics and no advantage claim | None. | None. |
| C3 Noise, transpilation and error analysis | Met | README.md, “Noise, transpilation, and error analysis”; root stage2_selection.json and stage3_simulation.json | No IBM hardware run; this is stated plainly. | None; do not describe simulator results as hardware results. |
| C4 Setup and reproducibility | Met | README.md, “Setup and reproducibility”; results/RUN_INFO.md; pinned requirements | None after clean-environment install, tests, pipeline, and notebook checks. | None. |
| C5 Business impact | Met | README.md, “Business impact and recommendations” | None. | None. |
| D Video and presentation links | Human task | docs/VIDEO_SCRIPT.md; docs/SLIDE_OUTLINE.md | No recorded public video or published view-only deck link. | Record the video and publish the deck; paste public links in Devpost. |
| E Devpost fields | Human task | docs/DEVPOST_FORM.md | Team/member contact fields and publication URLs remain placeholders. | Owner fills names, emails, Discord handles, and links. |

## Verified file facts

- Stage 1 root CSV has 100 rows; deterministic rank order sorts probability descending and customer_id ascending for ties.
- Stage 2 root CSV includes the ten ranked candidates and a selected flag; six are selected. JSON selections match Stage 1 probabilities and the package result files.
- Stage 3 JSON preserves the existing ideal estimate and exact calculation and adds the fixed-seed Monte Carlo benchmark, noisy results, compiled-circuit diagnostics, and probability-uncertainty stress analysis.
- The workbook hash and mapping fallback are documented in README.md and data/WORKBOOK_SHA256.txt.
- QAOA optimum-mass result remains the best of 80 measured runs; the exact subset remains the submitted answer.
