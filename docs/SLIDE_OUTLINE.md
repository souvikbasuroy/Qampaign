# Presentation outline

## Slide 1 — Campaign decision problem
- 300 labeled training customers, 100 unlabeled test customers, 35 positive labels (11.67% base rate).
- Three stages: predict response, select six under workbook resource limits, simulate at least three responses.
- Test IDs are explicit in the packaged workbook; documented fallback exists for an ID-less copy.

## Slide 2 — Stage 1: train-only prediction
- Nested repeated stratified CV, five repeats x five folds; preprocessing and model selection inside training.
- Logistic metrics: ROC-AUC 0.619, PR-AUC 0.226, Precision@10 0.600, Recall@10 0.171, Brier 0.097, log loss 0.342.
- ZZ SVC ROC-AUC 0.608, PR-AUC 0.149; alignment 0.034 and noisy kernel heatmap.
- Explain 10 train positives among poutcome=success and the fragile rank 10/11 edge.

## Slide 3 — Stage 2: formulation and exact result
- Expected net value p * expected_deposit_value - contact_cost.
- Exactly six selected; workbook budget 25 and capacity 22.
- Exact enumeration of 210 subsets: C0081, C0031, C0079, C0222, C0087, C0256; objective 16.122, cost 13, capacity 11.
- 10-qubit Dicke state and XY mixer QAOA; best-of-80 optimum mass 37.9%, median 0.68%, random chance 1/210. Exact remains submitted.
- QAOA transpiled linear-topology diagnostic depth 5,376 / 1,733 CX: routing is costly for this small case.

## Slide 4 — Stage 3: circuit and benchmark
- Six RY(2 arcsin sqrt(p)) gates; state qubit-to-customer order and little-endian strings.
- Ideal 20k-shot estimate 0.1810; exact 0.181754; 1M-trial Monte Carlo 0.181689.
- Stage 3 circuit: six mapped qubits, depth 5, no two-qubit gates.

## Slide 5 — Noise and uncertainty
- Synthetic Aer noise: 0.20195 unmitigated, 0.19038 readout-mitigated; compare exact.
- 10,000 shared calibration-bin stress draws: same Stage 2 set in all draws; Stage 3 95% stress quantiles 0.0291-0.4899.
- State the group-level proxy limitation and that no hardware run occurred.

## Slide 6 — Business recommendation and limits
- Use the exact six as a shortlist for controlled validation.
- Only 35 positive labels, unknown test outcomes, fragile rank boundary, scenario-proxy constraints.
- No quantum-advantage claim; QAOA is a comparison, exact search is the submitted selection.

## Slide 7 — Reproducibility and compliance
- Seed 42, library pins, commands, root CSV/JSON artifacts, tests, MIT license.
- Stage 1 train-only fitting; same probabilities used downstream; no test tuning.
- Link to README technical report and result files.
