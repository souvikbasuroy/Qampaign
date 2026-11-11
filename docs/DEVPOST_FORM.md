# Devpost form draft

## Team name
Qampaign (confirm the official team name before submission).

## Member names
[Add each participating member's full name.]

## Email addresses
[Add each member's preferred submission email.]

## Discord handles
[Add each member's Discord handle, or state “not applicable” if the form permits.]

## Project summary
Qampaign is a three-stage quantum-classical customer campaign decision workflow: train-only cross-validated response prediction, constrained top-customer selection, and a six-qubit simulation of the probability of at least three responses. The exact classical optimizer is the submitted Stage 2 answer; QAOA is reported as an experimental comparison.

## Exact six selected customer IDs
C0081, C0031, C0079, C0222, C0087, C0256.

## Final estimated P(at least 3)
Ideal Aer estimate: 0.1810 from 20,000 shots (95% Wilson interval 0.1757 to 0.1864). Exact independent-probability calculation: 0.181754. One-million-trial classical Monte Carlo: 0.181689 (95% Wilson interval 0.180934 to 0.182446).

## Quantum specs
- SDKs: Qiskit 2.6.2, qiskit-machine-learning 0.9.1, qiskit-algorithms 0.4.0, qiskit-aer 0.18.2.
- Backends: exact statevector kernel for Stage 1; Qiskit AerSimulator for QAOA and Stage 3. A separate AerSimulator configuration uses a linear coupling map and synthetic noise for transpilation/error diagnostics.
- No IBM hardware or real backend was used.
- Logical/mapped physical qubits: Stage 1 kernel 4 statevector features; Stage 2 10/10; Stage 3 6/6.
- Stage 2 transpilation: optimization level 1, seed_transpiler 42, depth 5,376, 1,733 CX gates, 3,753 RZ, 2,168 SX, 147 X.
- Stage 3 transpilation: optimization level 1, seed_transpiler 42, depth 5, 12 RZ, 12 SX, no 2-qubit gates.
- Shots: Stage 2 8,192 per each of 80 settings; ideal Stage 3 20,000; noisy Stage 3 20,000; Monte Carlo 1,000,000.
- Seeds: master 42; deterministic CV/model offsets and QAOA restart formula documented in Qampaign/results/RUN_INFO.md.

## Compliance statement
All preprocessing, calibration, feature/model selection, and hyperparameter selection were performed using Stage1_Train only through nested/repeated stratified CV. Stage1_Test was used once for continuous probability predictions and was not used to tune or select. Stage 2 and Stage 3 consume the same Stage 1 probabilities. Required result files are present at the repository root. Seed and simulator/transpiler settings are recorded. Classical and quantum results are reported separately and no quantum-advantage claim is made.

## Video URL
[Paste the public 1-2 minute video link after recording.]

## Presentation deck URL
[Paste the view-only presentation link after publishing.]
