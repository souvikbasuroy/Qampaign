# QuantumSentinel Customer Campaign — Executive Report

## Executive results

Stage 1 uses only 300 labeled customers (35 positives; 11.67% base rate). Nested repeated stratified CV selects regularized logistic regression by log loss. Logistic ROC-AUC is 0.619 (repeated-CV 95% interval 0.577 to 0.661; OOF bootstrap 0.515 to 0.717), PR-AUC 0.226 (0.193 to 0.259; bootstrap 0.137 to 0.365), Precision@10 0.600, Recall@10 0.171, Brier 0.097, and log loss 0.342. Quantum ZZ SVC ROC-AUC is 0.608, PR-AUC 0.149, Brier 0.102, and log loss 0.354. Kernel-target alignment is 0.034 and the heatmap is noisy/diffuse. Classical logistic performs better on PR-AUC and probability quality; no quantum advantage is claimed. Repeated-CV intervals understate uncertainty because folds overlap.

Precision@10 is 0.600 with zero repeat spread because the same ten customers rank first each repeat. Those are all ten training poutcome=success customers; six subscribed, so this result is effectively one feature. Precision@20 is 0.310 and Precision@30 is 0.213.

Only three test customers have clearly higher probabilities (0.3983, 0.3719, 0.3323). Ranks 4-10 are approximately 0.153-0.168. Rank 11 C0397 (0.1515) is 0.0013 below rank 10 C0256 (0.1528). The rank swap changes the exact six and objective from 16.122 to 17.306; it is a robustness note only.

## Selection and simulation

Exact Stage 2 selects C0081, C0031, C0079, C0222, C0087, C0256; objective 16.122; cost 13/25; capacity 11/22. All 210 combinations are feasible; maximum cost is 18 and maximum capacity is 17, so both limits are inactive for this pool. The exact optimum is the submitted answer.

The best measured QAOA result among 80 settings has optimum mass 3,105/8,192 = 37.9%, versus random chance 1/210 = 0.48%. Median mass is 0.68%, and 3/80 runs reached at least 10%. Best settings are COBYLA, p=4, restart 4, optimizer seed 40046, measurement seed 42. QAOA matches exact in this run, but 210 combinations are trivial to enumerate and do not establish quantum advantage.

Stage 3 estimates P(at least 3 of 6) as 0.1810 (20,000 shots; 95% Wilson interval 0.1757 to 0.1864), versus exact 0.181754. One million independent Bernoulli trials with seed 42 estimate 0.181689 (95% Wilson interval 0.180934 to 0.182446).

## Circuit and noise diagnostics

Stage 2's best-of-80 QAOA circuit is transpiled on a simulated linear coupling map with optimization level 1 and seed_transpiler 42: 10 logical/10 mapped physical qubits, depth 5,376, 1,733 CX, 3,753 RZ, 2,168 SX, 147 X. Stage 3 uses 6 logical/6 mapped physical qubits, depth 5, 12 RZ, 12 SX, and no two-qubit gates. No IBM backend or physical hardware was used.

With synthetic Aer noise (one-qubit depolarizing 0.005, CX depolarizing 0.02, readout assignment errors 0.02/0.03), Stage 3 gives 0.20195 unmitigated and 0.19038 after tensor-product readout assignment inversion. Absolute errors from exact are 0.02020 and 0.00863. The mitigation interval accounts for shot sampling, not calibration uncertainty.

A 10,000-draw shared additive perturbation from the top-decile OOF calibration range keeps the submitted six as the exact Stage 2 optimum in all draws. Stage 3 exact probability stress quantiles (2.5/50/97.5%) are 0.0291/0.2079/0.4899. This is group-level, not customer-specific uncertainty.

## Provenance and limits

The organizer's original workbook description says Stage1_Test lacks IDs, while the packaged workbook has explicit test IDs validated by the loader. The optional id_mapping setting remains supported; any row-order fallback is marked PROVISIONAL. Test labels are unknown. Only 35 positives support prediction; resource fields are workbook scenarios, not measured business constraints. Treat the chosen six as a shortlist for controlled validation, not a guaranteed campaign result.

Pinned versions, seeds, commands, and full result files are documented in README.md and Qampaign/results/RUN_INFO.md.
