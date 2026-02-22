# QuantumGrid — Quantum-Classical Hybrid for Energy Grid Optimisation

![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.44-6C3483)
![OR-Tools](https://img.shields.io/badge/OR--Tools-9.15-4285F4?logo=google&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-130%2B_passed-brightgreen)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> Variational quantum circuits (QAOA, VQE) applied to unit commitment on
> synthetic European grid data, benchmarked against classical MILP, simulated
> annealing, and greedy dispatch.

---

## Overview

The **unit commitment** (UC) problem — which generators to turn on/off over a
planning horizon to meet demand at minimum cost — is NP-hard and central to
energy grid operations. QuantumGrid formulates UC as a **QUBO** (Quadratic
Unconstrained Binary Optimisation), maps it to an Ising Hamiltonian, and solves
it with variational quantum algorithms on a statevector simulator.

**What this project demonstrates:**

1. Full QUBO encoding pipeline: generator fleet -> UC MILP -> QUBO matrix -> Ising Hamiltonian
2. QAOA solver (PennyLane `default.qubit`, `ApproxTimeEvolution`, COBYLA)
3. VQE solver (hardware-efficient Ry/Rz + CNOT ansatz)
4. Classical baselines: MILP (OR-Tools SCIP), simulated annealing, greedy merit-order
5. Penalty-weight tuning (binary search for lambda)
6. Scaling analysis and matplotlib visualisation suite

## Quick Start

```bash
# Clone
git clone https://github.com/ajliouat/quantumgrid.git && cd quantumgrid

# Create venv
python3 -m venv .venv && source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Run tests (130+ tests, ~2 min)
pytest -v --timeout=120

# Run demo
python demo.py
```

## Project Structure

```
quantumgrid/
    pyproject.toml
    demo.py                           # End-to-end demo (all solvers)
    data/
        download_entsoe.py            # ENTSO-E API stubs + synthetic fallback
        preprocess.py                 # Normalise, resample, horizon extraction
        generators.py                 # Generator/Fleet dataclasses, fleet builders
    formulation/
        unit_commitment.py            # Classical MILP via OR-Tools (SCIP/CBC)
        qubo_encoding.py              # UC -> QUBO -> Ising conversion
    quantum/
        cost_hamiltonian.py           # Ising -> PennyLane Hamiltonian, X-mixer
        qaoa_solver.py                # QAOA circuit + COBYLA optimiser
        vqe_solver.py                 # VQE with hardware-efficient ansatz
    classical/
        milp_solver.py                # Thin wrapper around UC MILP
        baselines.py                  # Simulated annealing + greedy dispatch
    benchmarks/
        penalty_tuning.py             # Binary search for lambda, sensitivity sweep
        scaling.py                    # Scaling study (classical + quantum)
    visualization/
        plots.py                      # Convergence, scaling, dispatch, heatmap plots
    tests/
        test_v100.py                  # Data + generators (28 tests)
        test_v101.py                  # MILP solver (16 tests)
        test_v102.py                  # QUBO encoding (20 tests)
        test_v103.py                  # QAOA circuit (9 tests)
        test_v104.py                  # VQE solver (9 tests)
        test_v105.py                  # Classical baselines (14 tests)
        test_v106.py                  # Penalty tuning (12 tests)
        test_v107.py                  # Scaling analysis (8 tests)
        test_v108.py                  # Visualisation (12 tests)
```

## Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Quantum simulation | PennyLane 0.44 `default.qubit` | Statevector QAOA/VQE |
| Classical optimiser | SciPy COBYLA | Variational parameter loop |
| MILP solver | Google OR-Tools (SCIP) | Exact UC baseline |
| Data generation | NumPy + pandas | Synthetic load/generation profiles |
| Visualisation | Matplotlib (Agg) | Convergence, dispatch, scaling plots |

## Key Results

| Problem Size | MILP | Greedy | SA | QAOA (p=1) | VQE |
|:------------|-----:|-------:|---:|-----------:|----:|
| 4 gen x 6 h (24 qubits) | Optimal | Fast, ~10% gap | Good convergence | Runs, quality varies | Expensive |
| 2 gen x 3 h (6 qubits) | Optimal | Exact | Near-optimal | Good | Good |

- MILP solves small instances in seconds; quantum solvers take minutes
- QAOA with p=1 on 24 qubits completes in ~100s on CPU
- VQE scales poorly (48 parameters for 24 qubits vs 2 for QAOA)
- Penalty tuning is critical for QUBO solution quality

## References

- Farhi et al., *Quantum Approximate Optimization Algorithm*, arXiv:1411.4028
- Peruzzo et al., *A variational eigenvalue solver*, arXiv:1304.3061
- Ajagekar et al., *Quantum Computing for Energy Systems*, arXiv:1906.09032
- Saravanan et al., *Unit Commitment: A Survey*, Renew. Sust. Energy Rev., 2013

## License

Apache 2.0
