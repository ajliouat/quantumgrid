"""VQE solver for the UC problem.

Variational Quantum Eigensolver with a hardware-efficient ansatz:
  - Ry/Rz single-qubit rotations
  - CNOT entangling layers (linear connectivity)
  - COBYLA or parameter-shift gradient optimizer
  - Same QUBO/Ising pipeline as QAOA
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from data.generators import GeneratorFleet
from formulation.qubo_encoding import (
    decode_solution,
    evaluate_qubo,
    qubo_to_ising,
    uc_to_qubo,
)
from quantum.cost_hamiltonian import ising_to_pennylane


@dataclass
class VQEResult:
    """Container for VQE solver output."""

    best_bitstring: np.ndarray
    best_cost: float
    schedule: np.ndarray
    dispatch: np.ndarray
    convergence: List[float]
    n_iterations: int
    solve_time_s: float
    optimal_params: np.ndarray
    n_layers: int
    ansatz: str


def _hardware_efficient_ansatz(params: np.ndarray, n_qubits: int, n_layers: int):
    """Apply hardware-efficient ansatz in-place.

    Each layer: Ry(theta) + Rz(phi) on each qubit, then CNOT ladder.
    params shape: (n_layers, n_qubits, 2)
    """
    p = params.reshape(n_layers, n_qubits, 2)
    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RY(p[layer, q, 0], wires=q)
            qml.RZ(p[layer, q, 1], wires=q)
        # Linear CNOT entanglement
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])


def build_vqe_circuit(
    cost_h: qml.Hamiltonian,
    n_qubits: int,
    n_layers: int,
):
    """Return QNodes for VQE cost evaluation and measurement probabilities.

    Args:
        cost_h: Cost Hamiltonian.
        n_qubits: Number of qubits.
        n_layers: Ansatz depth.

    Returns:
        (cost_qnode, probs_qnode)
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost_fn(params):
        _hardware_efficient_ansatz(params, n_qubits, n_layers)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def probs_fn(params):
        _hardware_efficient_ansatz(params, n_qubits, n_layers)
        return qml.probs(wires=range(n_qubits))

    return cost_fn, probs_fn


def solve_vqe(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    n_layers: int = 2,
    max_iterations: int = 100,
    lambda_demand: float = 100.0,
    lambda_reserve: float = 50.0,
    reserve_fraction: float = 0.10,
    seed: int = 42,
) -> VQEResult:
    """Run VQE on the UC problem.

    Args:
        fleet: Generator fleet.
        demand: (T,) demand array.
        n_layers: Ansatz depth.
        max_iterations: Max optimizer iterations.
        lambda_demand: Demand penalty weight.
        lambda_reserve: Reserve penalty weight.
        reserve_fraction: Reserve margin fraction.
        seed: Random seed.

    Returns:
        VQEResult.
    """
    t0 = perf_counter()
    rng = np.random.default_rng(seed)

    Q, meta = uc_to_qubo(fleet, demand, lambda_demand, lambda_reserve, reserve_fraction)
    J, h, offset = qubo_to_ising(Q)
    n_qubits = meta["n_qubits"]

    cost_h = ising_to_pennylane(J, h, offset)
    cost_fn, probs_fn = build_vqe_circuit(cost_h, n_qubits, n_layers)

    # Random initial parameters: (n_layers * n_qubits * 2)
    n_params = n_layers * n_qubits * 2
    init_params = rng.uniform(-np.pi, np.pi, n_params)

    convergence: List[float] = []

    def objective(params):
        val = float(cost_fn(params))
        convergence.append(val)
        return val

    result = minimize(
        objective,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )

    optimal_params = result.x

    # Extract best bitstring
    probs = np.array(probs_fn(optimal_params))
    best_idx = int(np.argmax(probs))
    best_bitstring = np.array(
        [(best_idx >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)],
        dtype=float,
    )
    best_cost = evaluate_qubo(Q, best_bitstring)
    schedule, dispatch = decode_solution(best_bitstring, fleet, len(demand))

    solve_time = perf_counter() - t0

    return VQEResult(
        best_bitstring=best_bitstring,
        best_cost=best_cost,
        schedule=schedule,
        dispatch=dispatch,
        convergence=convergence,
        n_iterations=len(convergence),
        solve_time_s=solve_time,
        optimal_params=optimal_params,
        n_layers=n_layers,
        ansatz="hardware_efficient_ry_rz",
    )
