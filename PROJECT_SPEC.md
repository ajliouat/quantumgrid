# QuantumGrid — Technical Specification

## 1. Problem Statement

Unit commitment (UC) is the problem of scheduling power generators on/off over a time horizon to meet forecasted demand at minimum cost, subject to engineering constraints. It's NP-hard (combinatorial explosion: N generators × T timesteps × binary decisions), making it a natural candidate for quantum optimization heuristics.

**This project:** Formulate UC as a QUBO problem, solve with QAOA and VQE, and rigorously compare against classical MILP and heuristics on real European grid data.

## 2. Data Source: ENTSO-E

### API Access
- Free registration at [transparency.entsoe.eu](https://transparency.entsoe.eu/)
- REST API with XML responses
- Data types: actual load, day-ahead forecast, generation by fuel type, installed capacity

### Data We Need
| Field | API Endpoint | Resolution |
|-------|-------------|-----------|
| Load forecast | A65 (Day-ahead total load forecast) | Hourly |
| Actual generation | A75 (Actual generation per type) | Hourly |
| Installed capacity | A68 (Installed generation capacity) | Annual |

**Target region:** France (large nuclear + renewable mix) or Germany (high renewable penetration)
**Time range:** 1 year of hourly data (8760 hours)

### Generator Fleet Model
Since ENTSO-E doesn't publish individual generator data, we construct a synthetic fleet that matches aggregate statistics:
- N generators with: capacity (MW), fuel cost ($/MWh), startup cost ($), min up/down time (h), ramp rate (MW/h)
- Calibrated so total fleet capacity ≈ country's installed capacity
- Fuel costs from [IEA World Energy Outlook](https://www.iea.org/topics/world-energy-outlook)

## 3. Mathematical Formulation

### 3.1 Classical UC (MILP)

**Variables:**
- x_{i,t} ∈ {0,1}: generator i on/off at time t
- p_{i,t} ∈ ℝ: power output of generator i at time t

**Objective:**
```
min Σ_t Σ_i [c_i · p_{i,t} + SU_i · startup_{i,t} + SD_i · shutdown_{i,t}]
```

**Constraints:**
```
Σ_i p_{i,t} ≥ D_t                     (demand satisfaction)
P_min_i · x_{i,t} ≤ p_{i,t} ≤ P_max_i · x_{i,t}  (capacity limits)
|p_{i,t} - p_{i,t-1}| ≤ R_i           (ramp limits)
Σ_i P_max_i · x_{i,t} ≥ D_t + Reserve (reserve margin)
```

### 3.2 QUBO Encoding for QAOA

**Step 1:** Discretize power output into K levels per generator:
```
p_{i,t} ≈ Σ_k 2^k · δ_{i,t,k} · ΔP_i    (binary encoding)
```

**Step 2:** Convert constraints to penalties:
```
H_cost = Σ costs (quadratic in binary variables)
H_demand = λ₁ · Σ_t (Σ_i p_{i,t} - D_t)²
H_ramp = λ₂ · Σ ramp violations²
H = H_cost + H_demand + H_ramp
```

**Step 3:** Map to Ising Hamiltonian via z_i = 2·x_i - 1

**Qubit count:** N generators × T timesteps × K precision bits
- Example: 6 generators × 6 hours × 3 bits = 108 qubits (too many for exact simulation)
- Reduced: 4 generators × 6 hours × 1 bit (on/off only) = 24 qubits (feasible)

### 3.3 VQE for Continuous OPF

VQE with hardware-efficient ansatz:
- Parameterized circuit with p layers of Ry, Rz rotations + CNOT entanglement
- Optimization via COBYLA (gradient-free) or parameter-shift rule (gradient-based)
- Expectation value of cost Hamiltonian is the objective

## 4. Implementation Details

### QAOA Circuit (PennyLane)

```python
# Pseudocode structure
def qaoa_circuit(gammas, betas, H_cost, H_mixer, n_layers):
    # Initial state: equal superposition
    for qubit in range(n_qubits):
        qml.Hadamard(wires=qubit)
    
    # Alternating layers
    for layer in range(n_layers):
        # Cost unitary
        qml.ApproxTimeEvolution(H_cost, gammas[layer], 1)
        # Mixer unitary  
        for qubit in range(n_qubits):
            qml.RX(2 * betas[layer], wires=qubit)
    
    return qml.expval(H_cost)
```

### Penalty Weight Selection

Critical and under-documented in quantum optimization literature:
1. Start with λ = max(cost coefficients)
2. Binary search: increase λ until constraint satisfaction > 95%
3. Report sensitivity analysis: cost vs constraint violation vs λ

### Scaling Analysis

Run each solver on problem instances of increasing size:
- N = 4, 6, 8, 10, 12, 14, 16, 18, 20 generators
- T = 6, 12, 24 hours
- For QAOA: layers p = 1, 2, 3, 5
- Record: solution cost, constraint violations, wall-clock time, convergence iterations

## 5. Baselines

| Solver | Type | Optimality | Scalability |
|--------|------|-----------|-------------|
| MILP (OR-Tools) | Exact | Optimal | Polynomial (with cuts) up to ~100 generators |
| Simulated Annealing | Heuristic | Near-optimal | Scales to 1000s of generators |
| Greedy Priority List | Heuristic | Suboptimal | Linear in N |
| QAOA (this project) | Quantum heuristic | Unknown | Limited by qubit count |
| VQE (this project) | Quantum heuristic | Unknown | Limited by qubit count |

## 6. Success Criteria

| Metric | Threshold |
|--------|-----------|
| QAOA matches MILP cost on N≤6 instances | Within 5% |
| Scaling curve clearly shows degradation point | ✓ |
| Constraint satisfaction analysis completed | ✓ |
| Convergence plots for all QAOA/VQE runs | ✓ |
| Real ENTSO-E data used and documented | ✓ |
| Honest discussion of quantum limitations | ✓ |

## 7. Timeline

| Week | Milestone |
|------|-----------|
| 1 | ENTSO-E API access. Download & preprocess load data. Build generator fleet model. |
| 2 | Implement classical MILP solver. Verify on small instances. |
| 3 | QUBO encoding. Implement QAOA circuit in PennyLane. Test on 4-generator instance. |
| 4 | VQE implementation. Ansatz experiments. |
| 5 | Simulated annealing and greedy baselines. |
| 6 | Scaling analysis across problem sizes. |
| 7 | Penalty weight sensitivity analysis. Convergence plots. |
| 8 | Full benchmark. Circuit diagrams. README with real results. Blog post. |
