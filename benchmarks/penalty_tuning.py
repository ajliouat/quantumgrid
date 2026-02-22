"""Penalty tuning utilities for QUBO formulations.

Provides:
  - Binary search for minimum lambda that satisfies demand constraints
  - Sensitivity analysis over a range of penalty weights
  - Constraint satisfaction checker for QUBO solutions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from data.generators import GeneratorFleet
from formulation.qubo_encoding import decode_solution, evaluate_qubo, uc_to_qubo


@dataclass
class TuningResult:
    """Result of penalty-weight tuning."""

    optimal_lambda_demand: float
    optimal_lambda_reserve: float
    best_cost: float
    constraint_satisfaction: float  # fraction of timesteps with demand met
    search_history: List[Dict]


@dataclass
class SensitivityPoint:
    """One evaluation point in a sensitivity sweep."""

    lambda_demand: float
    lambda_reserve: float
    qubo_cost: float
    fuel_cost: float
    demand_met_fraction: float
    reserve_met_fraction: float


def check_constraints(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    schedule: np.ndarray,
    dispatch: np.ndarray,
    reserve_fraction: float = 0.10,
    tolerance: float = 0.01,
) -> Dict:
    """Check constraint satisfaction for a decoded QUBO solution.

    Returns dict with:
      - demand_met: (T,) bool
      - reserve_met: (T,) bool
      - demand_gap: (T,) shortfall in MW
      - reserve_gap: (T,) shortfall in MW
      - demand_met_fraction: scalar
      - reserve_met_fraction: scalar
    """
    caps = fleet.capacity_vector()
    T = len(demand)

    gen_total = dispatch.sum(axis=0)
    online_cap = (schedule * caps[:, np.newaxis]).sum(axis=0)

    demand_gap = np.maximum(demand - gen_total, 0)
    demand_met = demand_gap <= tolerance * demand

    reserve_target = demand * (1 + reserve_fraction)
    reserve_gap = np.maximum(reserve_target - online_cap, 0)
    reserve_met = reserve_gap <= tolerance * reserve_target

    return {
        "demand_met": demand_met,
        "reserve_met": reserve_met,
        "demand_gap": demand_gap,
        "reserve_gap": reserve_gap,
        "demand_met_fraction": float(demand_met.mean()),
        "reserve_met_fraction": float(reserve_met.mean()),
    }


def _fuel_cost(fleet: GeneratorFleet, schedule: np.ndarray, dispatch: np.ndarray) -> float:
    """Compute raw fuel + startup cost (excluding penalties)."""
    N, T = schedule.shape
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


def binary_search_lambda(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    solution_fn,
    lambda_lo: float = 1.0,
    lambda_hi: float = 1000.0,
    max_steps: int = 10,
    target_satisfaction: float = 0.95,
    reserve_fraction: float = 0.10,
) -> TuningResult:
    """Binary search for minimum lambda_demand that achieves target constraint satisfaction.

    Args:
        fleet: Generator fleet.
        demand: (T,) demand array.
        solution_fn: Callable(fleet, demand, lambda_demand, lambda_reserve) -> bitstring.
        lambda_lo: Lower bound.
        lambda_hi: Upper bound.
        max_steps: Number of binary search steps.
        target_satisfaction: Required fraction of timesteps with demand met.
        reserve_fraction: Reserve margin.

    Returns:
        TuningResult with best lambda and satisfaction level.
    """
    history: List[Dict] = []
    best_lambda = lambda_hi
    best_cost = float("inf")
    best_sat = 0.0

    for step in range(max_steps):
        lam = (lambda_lo + lambda_hi) / 2
        lam_reserve = lam * 0.5  # heuristic: reserve penalty = 50% of demand

        bitstring = solution_fn(fleet, demand, lam, lam_reserve)
        n_timesteps = len(demand)
        schedule, dispatch = decode_solution(bitstring, fleet, n_timesteps)

        Q, _ = uc_to_qubo(fleet, demand, lam, lam_reserve, reserve_fraction)
        qubo_cost = evaluate_qubo(Q, bitstring)
        fuel = _fuel_cost(fleet, schedule, dispatch)
        check = check_constraints(fleet, demand, schedule, dispatch, reserve_fraction)

        sat = check["demand_met_fraction"]
        history.append({
            "step": step,
            "lambda_demand": lam,
            "lambda_reserve": lam_reserve,
            "qubo_cost": qubo_cost,
            "fuel_cost": fuel,
            "demand_sat": sat,
            "reserve_sat": check["reserve_met_fraction"],
        })

        if sat >= target_satisfaction:
            lambda_hi = lam
            if fuel < best_cost:
                best_cost = fuel
                best_lambda = lam
                best_sat = sat
        else:
            lambda_lo = lam

    return TuningResult(
        optimal_lambda_demand=best_lambda,
        optimal_lambda_reserve=best_lambda * 0.5,
        best_cost=best_cost,
        constraint_satisfaction=best_sat,
        search_history=history,
    )


def sensitivity_sweep(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    solution_fn,
    lambda_values: np.ndarray,
    reserve_fraction: float = 0.10,
) -> List[SensitivityPoint]:
    """Sweep over a range of penalty weights and record metrics.

    Args:
        fleet: Generator fleet.
        demand: (T,) demand array.
        solution_fn: Callable(fleet, demand, lambda_demand, lambda_reserve) -> bitstring.
        lambda_values: Array of lambda_demand values to test.
        reserve_fraction: Reserve margin.

    Returns:
        List of SensitivityPoint.
    """
    results: List[SensitivityPoint] = []
    n_timesteps = len(demand)

    for lam in lambda_values:
        lam_res = lam * 0.5
        bitstring = solution_fn(fleet, demand, lam, lam_res)
        schedule, dispatch = decode_solution(bitstring, fleet, n_timesteps)

        Q, _ = uc_to_qubo(fleet, demand, lam, lam_res, reserve_fraction)
        qubo_cost = evaluate_qubo(Q, bitstring)
        fuel = _fuel_cost(fleet, schedule, dispatch)
        check = check_constraints(fleet, demand, schedule, dispatch, reserve_fraction)

        results.append(SensitivityPoint(
            lambda_demand=lam,
            lambda_reserve=lam_res,
            qubo_cost=qubo_cost,
            fuel_cost=fuel,
            demand_met_fraction=check["demand_met_fraction"],
            reserve_met_fraction=check["reserve_met_fraction"],
        ))

    return results
