# QuantumGrid — Quantum-Classical Hybrid for Energy Grid Optimization

**Quantum AI × Energy Systems**

> Variational quantum circuits (VQE, QAOA) applied to unit commitment and optimal power flow on real European grid data, benchmarked against classical solvers.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

The unit commitment problem (which power plants to turn on/off over a planning horizon to meet demand at minimum cost) is NP-hard and central to energy grid operations. Classical solvers (MILP via CPLEX/Gurobi) work well for current grid sizes but struggle as renewable penetration increases variability and problem complexity.

Quantum optimization (QAOA, VQE) offers a potential speedup path for combinatorial problems. QuantumGrid implements both approaches on the same problem instances using real load/generation data from the European ENTSO-E Transparency Platform, providing an honest comparison of where quantum heuristics currently stand.

**This is not a "quantum supremacy" claim.** It's a rigorous benchmarking study that shows:
1. How to formulate grid optimization as a quantum problem
2. Where quantum approaches are competitive (small instances) 
3. Where they break down (scaling limits of current simulators)
4. The engineering of hybrid quantum-classical loops

## Problem Formulation

### Unit Commitment (QUBO formulation)

Given N generators over T time steps:
- **Binary decision variables:** x_{i,t} ∈ {0,1} — generator i on/off at time t
- **Objective:** Minimize total cost = fuel cost + startup cost + shutdown cost
- **Constraints:** Supply ≥ demand, ramp-up/down limits, min up/down times, reserve margin

**QUBO encoding:** Constraints embedded as penalty terms in the objective:
```
H = Σ costs(x) + λ₁·penalty_demand(x) + λ₂·penalty_ramp(x) + λ₃·penalty_reserve(x)
```

### Optimal Power Flow (continuous relaxation)

For VQE: encode generator dispatch levels as the expectation values of parameterized quantum circuits.

## Project Structure

```
quantumgrid/
├── README.md
├── PROJECT_SPEC.md
├── DEVELOPMENT_LOG.md
├── LICENSE
├── pyproject.toml
├── data/
│   ├── download_entsoe.py       # ENTSO-E API data fetcher
│   ├── preprocess.py            # Clean, normalize grid data
│   ├── generators.csv           # Generator fleet characteristics
│   └── sample/                  # Small sample data for testing
│       └── .gitkeep
├── formulation/
│   ├── unit_commitment.py       # Classical UC formulation (MILP)
│   ├── qubo_encoding.py         # UC → QUBO conversion
│   ├── optimal_power_flow.py    # OPF formulation
│   └── penalty_tuning.py        # Constraint penalty weight selection
├── quantum/
│   ├── qaoa.py                  # QAOA solver (PennyLane)
│   ├── vqe.py                   # VQE solver (PennyLane)
│   ├── ansatz.py                # Parameterized circuit ansätze
│   ├── cost_hamiltonian.py      # Ising Hamiltonian construction
│   └── optimizer.py             # Classical optimizer loop (COBYLA, L-BFGS)
├── classical/
│   ├── milp_solver.py           # CPLEX/OR-Tools MILP solver
│   ├── simulated_annealing.py   # SA baseline
│   └── greedy.py                # Greedy heuristic baseline
├── benchmarks/
│   ├── scaling_analysis.py      # Run solvers at N=4,6,8,...,20 generators
│   ├── time_horizon_sweep.py    # T=6,12,24,48 time steps
│   ├── compare_all.py           # Head-to-head comparison
│   └── results/
│       └── .gitkeep
├── visualization/
│   ├── convergence_plots.py     # VQE/QAOA cost function convergence
│   ├── scaling_plots.py         # Solution quality vs problem size
│   ├── circuit_diagrams.py      # Quantum circuit visualization
│   └── grid_dispatch.py         # Dispatch schedule visualization
├── tests/
│   ├── test_qubo.py
│   ├── test_qaoa.py
│   ├── test_vqe.py
│   └── test_milp.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── quantum_demo.ipynb       # Interactive QAOA/VQE demo
│   └── results_analysis.ipynb
└── .github/
    └── workflows/
        └── ci.yml
```

## Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| Quantum simulation | PennyLane + `default.qubit` | PyTorch integration, gradient support |
| Classical optimizer | SciPy (COBYLA, L-BFGS-B) | Standard for variational loops |
| MILP solver | Google OR-Tools / PuLP | Free, open-source (no Gurobi license needed) |
| Grid data | ENTSO-E Transparency Platform API | Real European load/generation data |
| Visualization | Matplotlib + PennyLane drawer | Circuit diagrams, convergence plots |

## Hardware

Everything runs on CPU — quantum simulation doesn't need GPUs.

| Task | Hardware | Estimated Time |
|------|----------|---------------|
| QAOA (N≤12 qubits) | Mac (CPU) | Minutes per instance |
| QAOA (N=16-20 qubits) | Mac (CPU) | Hours per instance |
| VQE (same scales) | Mac (CPU) | Similar |
| MILP solver | Mac (CPU) | Seconds (exact solution) |
| Full benchmark suite | Mac (CPU) | ~1 day total |

## Key Questions This Project Answers

1. **At what problem size does QAOA/VQE match MILP quality?** (Likely N≤8 generators)
2. **How does solution quality degrade as N grows beyond simulator capacity?** (Scaling curve)
3. **What's the optimization landscape like?** (Convergence plots, local minima analysis)
4. **How sensitive is QUBO performance to penalty weights?** (Constraint satisfaction analysis)
5. **Can VQE capture the continuous OPF problem better than discretized QAOA?**

## Benchmarks

_To be populated with real results:_

| Problem Size | MILP (exact) | Greedy | Sim. Annealing | QAOA | VQE | 
|-------------|-------------|--------|----------------|------|-----|
| 4 gen, 6h | $— | $— | $— | $— | $— |
| 8 gen, 12h | $— | $— | $— | $— | $— |
| 12 gen, 24h | $— | $— | $— | $— | $— |
| 16 gen, 24h | $— | $— | $— | $— | N/A |
| 20 gen, 48h | $— | $— | $— | N/A | N/A |

_Cost in $/MWh. QAOA/VQE report best of 10 random initializations. "N/A" = exceeds simulator capacity._

## References

- [QAOA: Quantum Approximate Optimization Algorithm (Farhi et al., 2014)](https://arxiv.org/abs/1411.4028)
- [VQE: A variational eigenvalue solver (Peruzzo et al., 2014)](https://arxiv.org/abs/1304.3061)
- [Quantum Computing for Energy Systems Optimization (Ajagekar et al., 2019)](https://arxiv.org/abs/1906.09032)
- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- [PennyLane Documentation](https://pennylane.ai/)
- [Unit Commitment: A Survey (Saravanan et al., 2013)](https://doi.org/10.1016/j.rser.2013.01.014)

## License

Apache 2.0
