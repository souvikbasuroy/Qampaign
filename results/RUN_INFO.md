# Run information

## Reproducibility

- Master seed: 42.
- Stage 1 outer repeated stratified CV: 5 repeats x 5 folds, split seed 42. Inner CV uses deterministic seeds offset by split index; final full-train search uses offset 9000. Calibration seeds use offsets 2000/4000 per outer split and 12000 for final fit. Bootstrap resampling uses seed 42 (quantum metric bootstrap uses 142).
- Model training randomness: fixed seed 42 for logistic solver, SVD reduction, and stochastic model steps. Test data has no labels and is not used to fit, calibrate, select, or tune.
- QAOA optimizer initial-state seeds: seed = 42 + p*10000 + optimizer_offset + restart; optimizer_offset is 0 for COBYLA and 5000 for SPSA. Best measured-mass run: p=4, COBYLA, restart 4, seed 40046.
- QAOA measurement seed: 42, 8,192 shots for each of 80 optimized configurations. Stage 3 ideal simulator: seed_simulator=42, 20,000 shots. Stage 3 noise simulator: seed_simulator=42, 20,000 shots. Monte Carlo seed: 42, 1,000,000 independent Bernoulli trials.
- Transpiler: optimization level 1, seed_transpiler=42 for Stage 2 and Stage 3 diagnostic circuits.
- Readout mitigation: tensor-product assignment-matrix inversion; readout calibration error is not included in its sampling-only interval.

## Library and runtime versions

Tested Python version: **3.13** (verified with Python 3.13.5).

Verified project environment versions:

- Python 3.13.5
- qiskit 2.6.2
- qiskit-machine-learning 0.9.1
- qiskit-algorithms 0.4.0
- qiskit-aer 0.18.2
- scikit-learn 1.9.1
- numpy 2.5.4
- pandas 3.1.1
- scipy 1.18.1
- mlflow 3.18.0
- matplotlib 3.11.2
- qiskit-optimization 0.7.0

All versions are pinned in Qampaign/requirements.txt.

## Quantum circuits and backends

| Use | Backend | Logical/mapped physical qubits | Shots | Notes |
|---|---|---:|---:|---|
| Stage 1 ZZ kernel | Exact statevector fidelity kernel | 4 statevector features / N/A | 0 | No hardware circuit execution |
| Stage 2 QAOA measured sweep | Qiskit AerSimulator | 10 / 10 decision qubits | 8,192 x 80 configs | Dicke weight-six state, XY mixer; exact answer is submitted |
| Stage 2 transpilation diagnostic | AerSimulator, linear coupling map | 10 / 10 used | No extra measurement run | Optimization level 1, depth 5,376, 1,733 CX |
| Stage 3 ideal | Qiskit AerSimulator | 6 / 6 | 20,000 | Ideal independent Bernoulli circuit |
| Stage 3 noisy | Qiskit AerSimulator, synthetic depolarizing/readout model | 6 / 6 | 20,000 | No real IBM backend or hardware used |
| Stage 3 Monte Carlo | NumPy independent Bernoulli | N/A | 1,000,000 trials | Seed 42 |

Transpiler statistics and logical-to-physical mappings are in the root stage2_selection.json and stage3_simulation.json. AerSimulator circuit registers can include unused ancilla slots; mapped physical-qubit counts report only used qubits.

## Submission artifacts

The pipeline writes root-level stage1_predictions.csv, stage2_selection.csv/.json, and stage3_simulation.csv/.json, while preserving package-local copies in Qampaign/results/. The Monte Carlo, circuit compilation/noise, error, and uncertainty-propagation details are additive fields and do not alter the previously reported stage results.
