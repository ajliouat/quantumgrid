# QuantumGrid — Development Log

> Build diary. Each version tagged and pushed.

---

## v1.0.0 — Scaffold + Data (28 tests)

- Created project skeleton: pyproject.toml, Dockerfile, CI, LICENSE, ROADMAP
- `data/download_entsoe.py`: ENTSO-E API stubs with synthetic fallback
  (load: 45 GW base + daily/weekly/seasonal; gen: nuclear/gas/wind/solar/hydro/coal)
- `data/preprocess.py`: normalise, resample, horizon extraction, temporal split
- `data/generators.py`: Generator dataclass, GeneratorFleet, build_fleet (mixed
  fuel types), build_small_fleet (quantum-scale)
- Python 3.14 venv with PennyLane 0.44, OR-Tools 9.15, scipy, pandas, matplotlib
- 28 tests passing in 0.42s

## v1.0.1 — Classical MILP Solver (16 tests)

- `formulation/unit_commitment.py`: Full UC MILP via OR-Tools pywraplp (SCIP/CBC)
  - Binary x[i,t] on/off, continuous p[i,t] dispatch, startup indicators
  - Constraints: demand balance, reserve margin, capacity bounds, startup logic
- `classical/milp_solver.py`: Thin wrapper
- UCResult dataclass with status, cost, schedule, dispatch, solve_time
- 16 tests passing in 14.85s

## v1.0.2 — QUBO Encoding & Ising Hamiltonian (20 tests)

- `formulation/qubo_encoding.py`: UC -> QUBO matrix conversion
  - H_cost (linear fuel), H_startup (approximate), H_demand (quadratic penalty),
    H_reserve (quadratic penalty)
  - QUBO -> Ising via x = (1-z)/2 substitution
  - evaluate_qubo, evaluate_ising, decode_solution
- `quantum/cost_hamiltonian.py`: Ising -> PennyLane Hamiltonian
  - ising_to_pennylane, mixer_hamiltonian (X-mixer), bitstring_energy
- Verified QUBO<->Ising energy equivalence over all 2^6 bitstrings
- 20 tests passing in 1.49s

## v1.0.3 — QAOA Circuit (9 tests)

- `quantum/qaoa_solver.py`: Full QAOA implementation
  - PennyLane default.qubit, ApproxTimeEvolution for cost + mixer unitaries
  - COBYLA classical outer loop
  - Probability-based bitstring extraction
- QAOAResult dataclass with convergence history
- 24-qubit smoke test passes in ~100s
- 9 tests passing in 106s

## v1.0.4 — VQE Solver (9 tests)

- `quantum/vqe_solver.py`: Hardware-efficient ansatz
  - Ry/Rz single-qubit rotations + CNOT linear entanglement
  - COBYLA optimiser, same pipeline as QAOA
- VQEResult dataclass
- Scales to 12 qubits in tests (24 qubits too slow for VQE due to O(n) params)
- 9 tests passing in 1.53s

## v1.0.5 — Classical Baselines (14 tests)

- `classical/baselines.py`:
  - Simulated annealing: bit-flip neighbourhood, geometric cooling, demand penalty
  - Greedy priority list: merit-order dispatch with reserve margin
- BaselineResult dataclass
- Helper functions: _compute_cost, _dispatch_for_schedule, _demand_violation
- 14 tests passing in 0.24s

## v1.0.6 — Penalty Tuning (12 tests)

- `benchmarks/penalty_tuning.py`:
  - binary_search_lambda: find minimum lambda for target constraint satisfaction
  - sensitivity_sweep: evaluate across lambda range
  - check_constraints: demand/reserve gap analysis
- TuningResult, SensitivityPoint dataclasses
- 12 tests passing in 0.31s

## v1.0.7 — Scaling Analysis (8 tests)

- `benchmarks/scaling.py`:
  - ScalingPoint/ScalingStudy data structures
  - run_classical_scaling: greedy + SA across N=[4,6,8,10], T=[6,12,24]
  - run_quantum_scaling: QAOA + VQE across N=[2,3,4], T=[3,4]
- 8 tests passing in 1.66s

## v1.0.8 — Visualisation (12 tests)

- `visualization/plots.py`:
  - plot_convergence: multi-solver convergence curves
  - plot_scaling: solve time vs problem size
  - plot_dispatch: stacked area generator dispatch vs demand
  - plot_schedule_heatmap: on/off binary heatmap
  - plot_sensitivity: dual-axis cost vs constraint satisfaction
- All use Agg backend for CI compatibility
- 12 tests passing in 0.96s

## v1.0.9 — Polish & Ship

- `demo.py`: end-to-end demo running all 5 solvers, saving plots to results/
- Updated README.md with final structure, stack, results table
- Updated DEVELOPMENT_LOG.md with full build history
- Final full test suite: 130+ tests all green
