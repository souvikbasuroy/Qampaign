# Qampaign - QuantumSentinel hackathon submission

## QUICKSTART

Tested Python version: **3.13** (verified with Python 3.13.5).

Run from the Qampaign directory:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests
    python main.py

The standalone Qampaign folder contains source, notebook, organizer workbook, results, tests, MIT license, settings, and pinned requirements. Running the pipeline also writes required judge-facing CSV/JSON files into the repository root.

## Data provenance and ID caveat

The organizer's original workbook description says Stage1_Test has no customer_id. The workbook packaged here contains 100 explicit test IDs; the loader validates and uses them directly. For an ID-less copy, set customer_project.id_mapping in settings.yaml to a CSV/XLSX with 1-based test_row and customer_id columns. Without workbook IDs or a mapping file, test row order is paired with sorted missing IDs and every stage marks this mapping PROVISIONAL.

Workbook SHA256: D2F8E63705AF2883C7618C463A6AC2F351119D59A8869D2677F8825148E9A709 (also data/WORKBOOK_SHA256.txt).

## Pipeline overview and methodology

Stage 1 compares regularized logistic regression and a ZZ quantum-kernel SVC using nested repeated stratified CV (outer 5 repeats x 5 folds, inner 3-fold tuning). Encoders, scalers, feature transforms, dimension reduction, feature/model selection, and angle scaling fit inside training folds. Quantum ranking uses decision_function; Platt calibration fits inside training folds. Test rows have no labels and are predicted once after model selection. Test data is never used for fitting, calibration, tuning, or selection. duration is excluded. Same Stage 1 probabilities flow into Stages 2 and 3.

Stage 2 joins the top ten with workbook economics and maximizes sum(p * expected_deposit_value - contact_cost). Exact search checks all 210 six-of-ten subsets and supplies the submitted result. QAOA is a comparison: 10 decision qubits, Dicke weight-six initial state, normalized objective, complete-graph XY mixer, p=1..4, 10 restarts for COBYLA and SPSA, 200 iterations, 8,192 shots per each of 80 optimized settings. Cardinality, budget, and capacity are checked after decoding. The alternative 20-qubit slack QUBO has 10 decision bits, five budget slack bits, and five capacity slack bits. Its energy is negative net value plus penalty weight 10,000 times squared exactly-six, budget-with-binary-slack, and capacity-with-binary-slack residuals. QUBO-to-Ising conversion uses x=(1-Z)/2. The Dicke/XY QAOA preserves cardinality and checks resource constraints after decoding. The alternative QUBO is checked against brute force.

Stage 3 prepares one qubit per selected customer with RY(2*arcsin(sqrt(p))). Success means at least 3 of 6 qubits are measured one. Qiskit count strings are little-endian: qubit 0 is the rightmost bit. The ideal simulator is compared with exact Poisson-binomial calculation, an independent Bernoulli Monte Carlo benchmark, and a synthetic-noise run.

## Classical vs quantum rigor

The nested-CV log-loss rule selects the final model. Classical and quantum results are separate; no quantum advantage is claimed. The chosen logistic model has ROC-AUC 0.619, PR-AUC 0.226, Precision@10 0.600, Recall@10 0.171, Brier 0.097, and log loss 0.342. The ZZ SVC has ROC-AUC 0.608, PR-AUC 0.149, and kernel-target alignment 0.034; its heatmap is noisy/diffuse. The quantum model loses on PR-AUC and probability quality.

Training has 35 positives among 300 customers (11.67% base rate). Intervals are 95%. Repeated-CV intervals are t-based over five outer OOF repeats; bootstrap intervals use 1,000 stratified resamples of pooled OOF predictions and labels. Repeated-CV intervals understate uncertainty because folds overlap.

| Logistic metric | Mean | Repeated-CV interval | OOF bootstrap interval |
|---|---:|---:|---:|
| ROC-AUC | 0.619 | 0.577 to 0.661 | 0.515 to 0.717 |
| PR-AUC | 0.226 | 0.193 to 0.259 | 0.137 to 0.365 |
| Precision@10 | 0.600; zero spread | no informative spread | 0.200 to 0.800 |
| Precision@20 | 0.310 | 0.282 to 0.338 | 0.100 to 0.550 |
| Precision@30 | 0.213 | 0.191 to 0.236 | 0.100 to 0.400 |
| Recall@10 | 0.171 | zero spread | 0.057 to 0.229 |
| Recall@20 | 0.177 | 0.161 to 0.193 | 0.057 to 0.314 |
| Recall@30 | 0.183 | 0.163 to 0.202 | 0.086 to 0.343 |
| Brier | 0.097 | 0.096 to 0.099 | 0.089 to 0.104 |
| Log loss | 0.342 | 0.336 to 0.348 | 0.317 to 0.363 |

| ZZ SVC metric | Mean | Repeated-CV interval | OOF bootstrap interval |
|---|---:|---:|---:|
| ROC-AUC | 0.608 | 0.582 to 0.633 | 0.535 to 0.704 |
| PR-AUC | 0.149 | 0.131 to 0.167 | 0.117 to 0.188 |
| Brier | 0.102 | 0.101 to 0.103 | 0.100 to 0.103 |
| Log loss | 0.354 | 0.349 to 0.359 | 0.343 to 0.361 |

Quantum raw decision-score fold ROC-AUC is 0.614 (0.576 to 0.653), compared with 0.607 (0.572 to 0.642) after Platt probabilities. Precision@10 is 0.600 with zero repeated-CV spread because the same ten customers rank first in every repeat. They are the ten training poutcome=success customers; six subscribed. Thus Precision@10 is essentially one feature.

## Top-ten fragility and calibration

Only three test customers have clearly higher probabilities: C0081 0.3983, C0031 0.3719, C0079 0.3323. Ranks 4-10 are about 0.153-0.168. Rank 11 C0397 at 0.1515 is only 0.0013 below rank 10 C0256 at 0.1528. Swapping rank 10 and 11 changes the exact selection to C0081, C0031, C0079, C0222, C0087, C0397; five of six IDs overlap and objective changes from 16.122 to 17.306. This is a robustness note only and does not change selection. Test top-ten probabilities are 0.153-0.398, within the OOF calibration range 0.052-0.536.

## Stage 2 exact selection and QAOA

Submitted exact selection: C0081, C0031, C0079, C0222, C0087, C0256; objective 16.122; contact cost 13/25; capacity 11/22. All 210 subsets are feasible. Maximum six-customer cost is 18 and capacity is 17, so limits are inactive for this pool.

The reported Stage 2 transpiled depth (5,376; 1,733 CX) includes generic state preparation of the weight-six Dicke state and routing of the all-to-all XY mixer on a linear topology, so a dedicated Dicke circuit would be shallower. The QAOA optimum mass is about 38% (best of 80 runs; varies slightly across machines), versus random feasible-subset chance 1/210 = 0.48%. Median mass is 0.68%; 3/80 settings reached at least 10%. Best settings: COBYLA, p=4, restart 4, optimizer seed 40046, simulator seed 42. QAOA matches the exact optimum on this run; both reported gaps are zero. Exact search is the submitted answer. This 210-combination problem does not demonstrate quantum advantage.

## Stage 3 simulation

The ideal estimate is 0.1810 from 20,000 shots (95% Wilson interval 0.1757 to 0.1864), versus exact 0.181754. A one-million-trial Bernoulli Monte Carlo with seed 42 gives 0.181689 (95% Wilson interval 0.180934 to 0.182446), absolute error 0.000065, relative error 0.000358. Sensitivity scenarios are in results/stage3_sensitivity.csv.

## Noise, transpilation, and error analysis

Transpilation uses Qiskit AerSimulator with a linear coupling map to make routing cost visible. This is a simulated topology, not a fake IBM device or hardware run. Optimization level 1 and seed_transpiler 42 are used; logical-to-physical mappings are identity mappings. Stage 2 uses 10 logical/10 mapped physical qubits, depth 5,376, 1,733 CX, 3,753 RZ, 2,168 SX, and 147 X gates (plus 10 measurements). Stage 3 uses 6 logical/6 mapped physical qubits, depth 5, 12 RZ, 12 SX, and no two-qubit gates (plus 6 measurements). Aer register width may include unused ancilla slots; physical count reports mapped qubits actually used.

The synthetic Stage 3 model has depolarizing probability 0.005 on 1-qubit X/SX gates, 0.02 on CX, and readout assignment errors 0.02/0.03. With 20,000 shots it estimates 0.20195 unmitigated (absolute error 0.02020) and 0.19038 after tensor-product readout assignment inversion (absolute error 0.00863). The mitigated interval accounts for shot sampling, not readout calibration uncertainty. No IBM backend or hardware was used.

To propagate Stage 1 probability uncertainty, 10,000 draws apply a shared additive offset sampled from the top-decile OOF calibration interval to all top-ten probabilities. The submitted Stage 2 six remain selected in 100% of draws. Stage 3 exact-probability 2.5th/50th/97.5th percentiles are 0.0291/0.2079/0.4899. This is a group-level stress-test proxy, not customer-specific test confidence intervals.

## Setup and reproducibility

From the Qampaign directory:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests
    python main.py

The notebook notebooks/customer_selection_workbook.ipynb runs the pipeline and reviews outputs. Settings are in settings.yaml. Package outputs and plots are in results/; root-level CSV/JSON files are the judge-facing outputs. MLflow logs locally to results/mlflow.db, created at runtime. See results/RUN_INFO.md for versions, seeds, backends, shots, and qubit counts. There is no cloud account requirement.

## Business impact and recommendations

Treat the selected six as a shortlist for controlled validation. The objective is 16.122 workbook scenario units, not guaranteed financial return. Only 35 positive labels are available; test outcomes are unknown; costs/capacity are scenario assumptions; and ranks 10/11 are fragile. Do not claim guaranteed campaign lift or quantum advantage.

## Compliance checklist

- [x] Train-only preprocessing, calibration, feature/model selection, and tuning.
- [x] Same Stage 1 probabilities feed Stage 2 and Stage 3.
- [x] No test-set tuning or calibration.
- [x] Classical and quantum results are separate; no quantum-advantage claim.
- [x] Required result files and schemas at repository root.
- [x] Seeds, versions, backends, shots, and qubit counts documented.
- [x] MIT license and pinned requirements included.
- [ ] Publish the GitHub repository and public video/deck links (human tasks).

## Quantum specs

| Stage | SDK/backend | Logical / mapped physical qubits | Transpiled depth / gates | Shots | Seeds |
|---|---|---:|---|---:|---|
| Stage 1 ZZ kernel | Qiskit 2.6.2 exact statevector | 4 / N/A | Exact kernel; no gate transpilation | 0 | 42 |
| Stage 2 QAOA | Qiskit AerSimulator, linear simulated topology | 10 / 10 | Depth 5,376; CX 1,733; RZ 3,753; SX 2,168; X 147 | 8,192 x 80 | derived optimizer seeds; simulator 42; transpiler 42 |
| Stage 3 ideal | Qiskit AerSimulator, linear simulated topology | 6 / 6 | Depth 5; RZ 12; SX 12; 2-qubit gates 0 | 20,000 | simulator 42; transpiler 42 |
| Stage 3 noisy | Qiskit AerSimulator synthetic noise | 6 / 6 | Same Stage 3 circuit | 20,000 | simulator 42; transpiler 42 |

No real IBM backend or physical hardware was used. The workbook hash and id_mapping option are documented above. The QAOA optimum mass is a best-of-80 result, not a typical-run guarantee.

## Submission result files

- Repository root: stage1_predictions.csv; stage2_selection.csv and stage2_selection.json; stage3_simulation.csv and stage3_simulation.json.
- Package review outputs and plots: results/.
- Review drafts: docs/VIDEO_SCRIPT.md, docs/SLIDE_OUTLINE.md, docs/DEVPOST_FORM.md, docs/REQUIREMENT_AUDIT.md.
