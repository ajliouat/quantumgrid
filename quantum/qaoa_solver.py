"""QAOA solver for the UC problem.

Implements the Quantum Approximate Optimization Algorithm using PennyLane:
  1. Build cost + mixer Hamiltonians from QUBO/Ising encoding.
  2. Parameterised QAOA circuit with p layers of (gamma, beta).
  3. Classical outer-loop optimizer (gradient-free).
  4. Sample best bitstring → decode to schedule/dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml

from data.generators import GeneratorFleet
from formulation.qubo_encoding import (
    decode_solution,
    evaluate_qubo,
    qubo_to_ising,
    uc_to_qubo,
)
from quantum.cost_hamiltonian import ising_to_pennylane, mixer_hamiltonian


@dataclass
class QAOAResult:
    """Container for QAOA solver output."""

    best_bitstring: np.ndarray
    best_cost: float
    schedule: np.ndarray
    dispatch: np.ndarray
    convergence: List[float]
    n_iterations: int
    solve_time_s: float
    optimal_params: np.ndarray
    n_layers: int


def build_qaoa_circuit(
    cost_h: qml.Hamiltonian,
    mixer_h: qml.Hamiltonian,
    n_qubits: int,
    n_layers: int,
):
    """Return a QNode that evaluates the QAOA cost expectation.

    Args:
        cost_h: Cost Hamiltonian (Ising).
        mixer_h: Mixer Hamiltonian (X-mixer).
        n_qubits: Number of qubits.
        n_layers: Number of QAOA layers p.

    Returns:
        (cost_qnode, probs_qnode): QNodes for cost and probability.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(params):
        """Evaluate <psi(gamma,beta)|H_cost|psi(gamma,beta)>."""
        gammas = params[:n_layers]
        betas = params[n_layers:]
        # Initial superposition
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        # QAOA layers
        for layer in range(n_layers):
            # Cost unitary: exp(-i * gamma * H_cost)
            qml.ApproxTimeEvolution(cost_h, gammas[layer], 1)
            # Mixer unitary: exp(-i * beta * H_mixer)
            qml.ApproxTimeEvolution(mixer_h, betas[layer], 1)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def probs_fn(params):
        """Return measurement probabilities over computational basis."""
        gammas = params[:n_layers]
        betas = params[n_layers:]
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for layer in range(n_layers):
            qml.ApproxTimeEvolution(cost_h, gammas[layer], 1)
            qml.ApproxTimeEvolution(mixer_h, betas[layer], 1)
        return qml.probs(wires=range(n_qubits))

    return cost_fn, probs_fn


def solve_qaoa(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    n_layers: int = 2,
    max_iterations: int = 80,
    lambda_demand: float = 100.0,
    lambda_reserve: float = 50.0,
    reserve_fraction: float = 0.10,
    seed: int = 42,
) -> QAOAResult:
    """Run QAOA on the UC problem.

    Args:
        fleet: Generator fleet.
        demand: (T,) demand array.
        n_layers: Number of QAOA layers p.
        max_iterations: Max optimizer iterations.
        lambda_demand: Demand penalty weight.
        lambda_reserve: Reserve penalty weight.
        reserve_fraction: Reserve margin fraction.
        seed: Random seed for initial parameters.

    Returns:
        QAOAResult with best solution found.
    """
    t0 = perf_counter()
    rng = np.random.default_rng(seed)

    # Build QUBO & Ising
    Q, meta = uc_to_qubo(fleet, demand, lambda_demand, lambda_reserve, reserve_fraction)
    J, h, offset = qubo_to_ising(Q)
    n_qubits = meta["n_qubits"]

    cost_h = ising_to_pennylane(J, h, offset)
    mix_h = mixer_hamiltonian(n_qubits)

    cost_fn, probs_fn = build_qaoa_circuit(cost_h, mix_h, n_qubits, n_layers)

    # Initial parameters: gamma ~ U(0, 2pi), beta ~ U(0, pi)
    init_params = np.concatenate([
        rng.uniform(0, 2 * np.pi, n_layers),
        rng.uniform(0, np.pi, n_layers),
    ])

    # COBYLA optimizer (gradient-free)
    convergence: List[float] = []

    def objective(params):
        val = float(cost_fn(params))
        convergence.append(val)
        return val

    from scipy.optimize import minimize

    result = minimize(
        objective,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )

    optimal_params = result.x

    # Extract best bitstring from measurement probabilities
    probs = np.array(probs_fn(optimal_params))
    best_idx = int(np.argmax(probs))
    best_bitstring = np.array(
        [(best_idx >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)],
        dtype=float,
    )

    # Evaluate QUBO cost for the best bitstring
    best_cost = evaluate_qubo(Q, best_bitstring)

    # Decode
    schedule, dispatch = decode_solution(best_bitstring, fleet, len(demand))

    solve_time = perf_counter() - t0

    return QAOAResult(
        best_bitstring=best_bitstring,
        best_cost=best_cost,
        schedule=schedule,
        dispatch=dispatch,
        convergence=convergence,
        n_iterations=len(convergence),
        solve_time_s=solve_time,
        optimal_params=optimal_params,
        n_layers=n_layers,
    )
