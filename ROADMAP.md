# QuantumGrid — Release Roadmap

## v1.0.0 — Scaffold, Data & Generator Fleet
## v1.0.1 — Classical MILP Solver
## v1.0.2 — QUBO Encoding
## v1.0.3 — QAOA Circuit (PennyLane)
## v1.0.4 — VQE Solver
## v1.0.5 — Classical Baselines (SA, Greedy)
## v1.0.6 — Penalty Tuning & Sensitivity
## v1.0.7 — Scaling Analysis
## v1.0.8 — Visualisation Suite
## v1.0.9 — Polish & Ship

---

## Progress Tracker

| Release | Status | Key Result |
|---------|--------|------------|
| v1.0.0 | ✅ Complete | Scaffold, data pipeline, generator fleet |
| v1.0.1 | ✅ Complete | MILP solver (OR-Tools SCIP), 16 tests |
| v1.0.2 | ✅ Complete | QUBO encoding + Ising mapping, 20 tests |
| v1.0.3 | ✅ Complete | QAOA solver (PennyLane), 9 tests |
| v1.0.4 | ✅ Complete | VQE solver (Ry/Rz + CNOT), 9 tests |
| v1.0.5 | ✅ Complete | SA + greedy baselines, 14 tests |
| v1.0.6 | ✅ Complete | Penalty tuning + sensitivity, 12 tests |
| v1.0.7 | ✅ Complete | Scaling analysis, 8 tests |
| v1.0.8 | ✅ Complete | Visualisation suite, 12 tests |
| v1.0.9 | ✅ Complete | Polish, README, blog post |

---

## Future Evolution

> These iterations are aspirational — they represent natural next steps for
> the project if development resumes. Not currently scheduled.

### v1.1.0 — QAOA Depth Sweep & Warm-Starting

**Goal:** Study the effect of QAOA depth p on solution quality and implement warm-start strategies.

- Sweep p from 1 to 8 on 6-qubit and 12-qubit instances
- Parameter transfer: use optimal p=k params to initialise p=k+1
- INTERP strategy (Interpolation from lower depth)
- Compare COBYLA vs L-BFGS-B vs ADAM for variational loop
- Track approximation ratio vs depth

### v1.2.0 — Noise Models & Error Mitigation

**Goal:** Simulate realistic quantum hardware noise and test error mitigation.

- PennyLane `default.mixed` backend with depolarizing noise
- Single-qubit gate error: p=0.001, two-qubit: p=0.01
- Zero-noise extrapolation (ZNE) via noise scaling
- Probabilistic error cancellation (PEC)
- Compare noisy vs ideal solution quality across problem sizes

### v1.3.0 — Constraint-Preserving Mixers

**Goal:** Replace the standard X-mixer with constraint-preserving alternatives.

- XY-mixer to restrict search to feasible subspace (Hadfield et al., 2019)
- Grover mixer for hard constraint handling
- Custom demand-preserving mixer
- Compare feasibility rates: standard X-mixer vs XY-mixer vs Grover

### v1.4.0 — Tensor Network Simulation

**Goal:** Push beyond statevector limits (~26 qubits) using tensor network methods.

- MPS (Matrix Product State) simulator for 30–50 qubit instances
- quimb or PennyLane `default.tensor` backend
- Bond dimension sweep: accuracy vs memory tradeoff
- Profile wall-clock time at 30, 40, 50 qubits

### v1.5.0 — Real Hardware Execution

**Goal:** Run QAOA on real quantum hardware (IBM Quantum or Amazon Braket).

- Transpile circuits for target backend topology
- Gate count and circuit depth analysis post-transpilation
- Shot budget study: 1K, 4K, 16K, 64K shots vs solution quality
- Error mitigation: M3 (Matrix-free Measurement Mitigation)

### v1.6.0 — Multi-Period Renewable Integration

**Goal:** Extend the UC model to include renewable generation and storage.

- Wind and solar generation profiles (synthetic + ENTSO-E historical)
- Battery storage as dispatchable resource
- Net demand = demand - renewables → dynamic QUBO size
- Stochastic UC with scenario sampling

---

*v1.0.0–v1.0.9 delivered. Future iterations begin when development resumes.*
