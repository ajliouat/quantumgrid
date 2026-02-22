"""Classical baseline solvers for the UC problem.

Provides two heuristics for comparison with quantum approaches:
  1. Simulated Annealing (SA) — stochastic local search on binary schedule
  2. Greedy Priority List — merit-order dispatch by fuel cost
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

import numpy as np

from data.generators import GeneratorFleet


@dataclass
class BaselineResult:
    """Result from a classical baseline solver."""

    schedule: np.ndarray
    dispatch: np.ndarray
    total_cost: float
    convergence: List[float]
    solve_time_s: float
    method: str


# ---------- helpers ----------
def _compute_cost(fleet: GeneratorFleet, schedule: np.ndarray, dispatch: np.ndarray) -> float:
    """Total cost = fuel cost + startup cost."""
    N, T = schedule.shape
    caps = fleet.capacity_vector()
    costs = fleet.cost_vector()
    su_costs = fleet.startup_cost_vector()
    total = 0.0
    for i in range(N):
        for t in range(T):
            if schedule[i, t]:
                total += costs[i] * dispatch[i, t]
                if t == 0 or not schedule[i, t - 1]:
                    total += su_costs[i]
    return total


def _dispatch_for_schedule(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    schedule: np.ndarray,
) -> np.ndarray:
    """Greedy economic dispatch for a given on/off schedule."""
    N, T = schedule.shape
    caps = fleet.capacity_vector()
    costs = fleet.cost_vector()
    mins = fleet.min_output_vector()
    dispatch = np.zeros((N, T))
    for t in range(T):
        on = np.where(schedule[:, t] == 1)[0]
        if len(on) == 0:
            continue
        # Set minimum outputs first
        for i in on:
            dispatch[i, t] = mins[i]
        remaining = demand[t] - dispatch[:, t].sum()
        # Fill by merit order (cheapest first)
        order = sorted(on, key=lambda i: costs[i])
        for i in order:
            if remaining <= 0:
                break
            room = caps[i] - dispatch[i, t]
            add = min(room, remaining)
            dispatch[i, t] += add
            remaining -= add
    return dispatch


def _demand_violation(dispatch: np.ndarray, demand: np.ndarray) -> float:
    """Sum of absolute demand shortfalls across timesteps."""
    return float(np.sum(np.maximum(demand - dispatch.sum(axis=0), 0)))


# ========== Simulated Annealing ==========
def solve_simulated_annealing(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    max_iterations: int = 2000,
    t_init: float = 100.0,
    t_min: float = 0.01,
    cooling: float = 0.995,
    demand_penalty: float = 1000.0,
    seed: int = 42,
) -> BaselineResult:
    """Simulated annealing on the binary schedule.

    Energy = total_cost + demand_penalty * demand_violation.
    Neighbourhood: flip one x_{i,t} bit.
    """
    t0 = perf_counter()
    rng = np.random.default_rng(seed)
    N = fleet.n_generators
    T = len(demand)

    # Initial: greedy schedule
    schedule = np.zeros((N, T), dtype=int)
    order = np.argsort(fleet.cost_vector())
    for t in range(T):
        total_cap = 0.0
        for i in order:
            schedule[i, t] = 1
            total_cap += fleet.capacity_vector()[i]
            if total_cap >= demand[t]:
                break

    dispatch = _dispatch_for_schedule(fleet, demand, schedule)
    cost = _compute_cost(fleet, schedule, dispatch)
    viol = _demand_violation(dispatch, demand)
    energy = cost + demand_penalty * viol

    best_schedule = schedule.copy()
    best_dispatch = dispatch.copy()
    best_energy = energy
    convergence: List[float] = [energy]

    temp = t_init
    for _ in range(max_iterations):
        # Flip random bit
        i, t_idx = rng.integers(N), rng.integers(T)
        new_schedule = schedule.copy()
        new_schedule[i, t_idx] = 1 - new_schedule[i, t_idx]
        new_dispatch = _dispatch_for_schedule(fleet, demand, new_schedule)
        new_cost = _compute_cost(fleet, new_schedule, new_dispatch)
        new_viol = _demand_violation(new_dispatch, demand)
        new_energy = new_cost + demand_penalty * new_viol

        delta = new_energy - energy
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            schedule = new_schedule
            dispatch = new_dispatch
            energy = new_energy
            if energy < best_energy:
                best_schedule = schedule.copy()
                best_dispatch = dispatch.copy()
                best_energy = energy

        temp *= cooling
        if temp < t_min:
            temp = t_min
        convergence.append(best_energy)

    solve_time = perf_counter() - t0
    total_cost = _compute_cost(fleet, best_schedule, best_dispatch)

    return BaselineResult(
        schedule=best_schedule,
        dispatch=best_dispatch,
        total_cost=total_cost,
        convergence=convergence,
        solve_time_s=solve_time,
        method="simulated_annealing",
    )


# ========== Greedy Priority List ==========
def solve_greedy(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    reserve_fraction: float = 0.10,
) -> BaselineResult:
    """Merit-order greedy dispatch.

    At each timestep, turn on cheapest generators until demand + reserve met.
    """
    t0 = perf_counter()
    N = fleet.n_generators
    T = len(demand)
    caps = fleet.capacity_vector()
    costs = fleet.cost_vector()

    schedule = np.zeros((N, T), dtype=int)
    order = np.argsort(costs)  # merit order

    for t in range(T):
        target = demand[t] * (1 + reserve_fraction)
        total_cap = 0.0
        for i in order:
            schedule[i, t] = 1
            total_cap += caps[i]
            if total_cap >= target:
                break

    dispatch = _dispatch_for_schedule(fleet, demand, schedule)
    total_cost = _compute_cost(fleet, schedule, dispatch)
    solve_time = perf_counter() - t0

    return BaselineResult(
        schedule=schedule,
        dispatch=dispatch,
        total_cost=total_cost,
        convergence=[total_cost],
        solve_time_s=solve_time,
        method="greedy_priority_list",
    )
