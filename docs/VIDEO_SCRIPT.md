# 90-second demo script

## 0:00-0:12 — Problem and controls
Show the workbook and project architecture. Say: “We predict response from 300 labeled customers, rank 100 unlabeled customers, select six under workbook constraints, and simulate the chance that at least three respond. All preprocessing and selection use training CV only; seed 42 is fixed.”

## 0:12-0:30 — Stage 1 prediction
Show the notebook and nested-CV comparison/calibration plot. Say: “The selected logistic model has ROC-AUC 0.619, PR-AUC 0.226, Precision@10 0.600, Recall@10 0.171, Brier 0.097, and log loss 0.342. It beats the ZZ kernel on probability quality; the kernel alignment is 0.034 and its matrix is noisy. We make no quantum-advantage claim.”

## 0:30-0:50 — Stage 2 optimization
Show the top-ten CSV, QAOA circuit/optimizer sweep, and exact-search result. Say: “Exact search checks all 210 subsets and selects C0081, C0031, C0079, C0222, C0087, and C0256 for objective 16.122, cost 13 of 25, capacity 11 of 22. QAOA's best of 80 runs puts 37.9% of measured mass on the exact optimum; exact search remains our submitted answer.”

## 0:50-1:08 — Stage 3 circuit and error
Show the six RY gates and the little-endian mapping. Say: “Each qubit is prepared with RY of twice arcsin square-root probability. The 20,000-shot ideal estimate is 0.1810; exact calculation is 0.181754; a million-trial classical Monte Carlo gives 0.181689. Under our stated synthetic Aer noise model, the estimate moves to 0.20195, or 0.19038 after readout correction. This is simulation, not hardware.”

## 1:08-1:30 — Recommendation and limits
Show the README quantum-specs table and root JSON artifacts. Say: “We recommend treating these six as a testable shortlist. Only 35 positives are labeled, the rank-10 boundary is fragile, and costs and capacity are scenario inputs. All result files, seeds, versions, and tests are included for review.”
