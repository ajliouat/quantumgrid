"""QUBO encoding for the Unit Commitment problem.

Converts the UC problem into a Quadratic Unconstrained Binary Optimization
(QUBO) formulation by embedding constraints as penalty terms:

  H = H_cost + lambda_demand * H_demand + lambda_reserve * H_reserve

Binary variables:
  x_{i,t} in {0,1} — generator i on/off at time t

The QUBO matrix Q is such that x^T Q x = H(x).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from data.generators import GeneratorFleet


def uc_to_qubo(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    lambda_demand: float = 100.0,
    lambda_reserve: float = 50.0,
    reserve_fraction: float = 0.10,
) -> Tuple[np.ndarray, Dict]:
    """Convert unit commitment to QUBO matrix.

    Binary decision: x_{i,t} = 1 if generator i is on at time t.
    When on, generator produces at full capacity (simplified for QUBO).

    Args:
        fleet: Generator fleet.
        demand: (T,) demand array in MW.
        lambda_demand: Penalty weight for demand constraint.
        lambda_reserve: Penalty weight for reserve constraint.
        reserve_fraction: Reserve margin fraction.

    Returns:
        Q: (n_qubits, n_qubits) QUBO matrix.
        metadata: dict with variable mapping and problem info.
    """
    N = fleet.n_generators
    T = len(demand)
    n_qubits = N * T

    caps = fleet.capacity_vector()
    costs = fleet.cost_vector()
    su_costs = fleet.startup_cost_vector()

    Q = np.zeros((n_qubits, n_qubits))

    def idx(i: int, t: int) -> int:
        """Map (generator, timestep) to qubit index."""
        return i * T + t

    # --- H_cost: linear fuel cost (when on, produce at capacity) ---
    for i in range(N):
        for t in range(T):
            q = idx(i, t)
            Q[q, q] += costs[i] * caps[i]

    # --- H_startup: startup cost (approximate) ---
    # startup[i,t] approx x[i,t] * (1 - x[i,t-1])
    # = x[i,t] - x[i,t] * x[i,t-1]
    for i in range(N):
        for t in range(T):
            q_t = idx(i, t)
            if t == 0:
                # First timestep: startup if on
                Q[q_t, q_t] += su_costs[i]
            else:
                q_prev = idx(i, t - 1)
                Q[q_t, q_t] += su_costs[i]
                # Interaction: -su_cost * x[i,t] * x[i,t-1]
                row, col = min(q_t, q_prev), max(q_t, q_prev)
                Q[row, col] -= su_costs[i]

    # --- H_demand: (sum_i cap_i * x_{i,t} - D_t)^2 ---
    for t in range(T):
        for i in range(N):
            qi = idx(i, t)
            # Linear: cap_i^2 - 2*cap_i*D_t
            Q[qi, qi] += lambda_demand * (caps[i] ** 2 - 2 * caps[i] * demand[t])
            # Quadratic: cap_i * cap_j
            for j in range(i + 1, N):
                qj = idx(j, t)
                row, col = min(qi, qj), max(qi, qj)
                Q[row, col] += lambda_demand * 2 * caps[i] * caps[j]

    # --- H_reserve: penalty if total online capacity < D_t * (1+r) ---
    # Same quadratic form with different target
    reserve_demand = demand * (1 + reserve_fraction)
    for t in range(T):
        for i in range(N):
            qi = idx(i, t)
            Q[qi, qi] += lambda_reserve * (caps[i] ** 2 - 2 * caps[i] * reserve_demand[t])
            for j in range(i + 1, N):
                qj = idx(j, t)
                row, col = min(qi, qj), max(qi, qj)
                Q[row, col] += lambda_reserve * 2 * caps[i] * caps[j]

    metadata = {
        "n_generators": N,
        "n_timesteps": T,
        "n_qubits": n_qubits,
        "variable_map": {(i, t): idx(i, t) for i in range(N) for t in range(T)},
    }

    return Q, metadata


def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO matrix to Ising model coefficients.

    Uses substitution x_i = (1 + z_i) / 2 where z_i in {-1, +1}.

    Returns:
        J: (n, n) coupling matrix (upper triangular).
        h: (n,) local field vector.
        offset: constant energy offset.
    """
    n = Q.shape[0]
    # Make Q upper triangular (merge Q[i,j] and Q[j,i] into upper)
    Q_upper = np.triu(Q) + np.triu(Q.T, k=1)

    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0.0

    # x_i = (1 - z_i)/2  (since z = 1 - 2x)
    # x_i * x_j = (1 - z_i)(1 - z_j)/4 = (1 - z_i - z_j + z_i*z_j)/4
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Q[i,i] * x_i = Q[i,i] * (1 - z_i)/2
                h[i] -= Q_upper[i, i] / 2
                offset += Q_upper[i, i] / 2
            else:
                # Q[i,j] * x_i * x_j = Q[i,j] * (1 - z_i - z_j + z_i*z_j)/4
                J[i, j] += Q_upper[i, j] / 4
                h[i] -= Q_upper[i, j] / 4
                h[j] -= Q_upper[i, j] / 4
                offset += Q_upper[i, j] / 4

    return J, h, offset


def evaluate_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluate QUBO objective for a binary vector x.

    Args:
        Q: (n, n) QUBO matrix.
        x: (n,) binary vector {0, 1}.

    Returns:
        Scalar objective value x^T Q x.
    """
    return float(x @ Q @ x)


def evaluate_ising(
    J: np.ndarray,
    h: np.ndarray,
    offset: float,
    z: np.ndarray,
) -> float:
    """Evaluate Ising energy for a spin configuration z.

    Args:
        J: (n, n) coupling matrix.
        h: (n,) local field.
        offset: constant offset.
        z: (n,) spin vector {-1, +1}.

    Returns:
        Energy value.
    """
    return float(z @ J @ z + h @ z + offset)


def decode_solution(
    x: np.ndarray,
    fleet: GeneratorFleet,
    n_timesteps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode QUBO solution to schedule and dispatch.

    Args:
        x: (n_qubits,) binary solution vector.
        fleet: Generator fleet.
        n_timesteps: Number of timesteps T.

    Returns:
        schedule: (N, T) binary on/off.
        dispatch: (N, T) power output (capacity when on, 0 when off).
    """
    N = fleet.n_generators
    T = n_timesteps
    caps = fleet.capacity_vector()

    schedule = x.reshape(N, T)
    dispatch = schedule * caps[:, np.newaxis]

    return schedule, dispatch
