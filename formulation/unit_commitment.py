"""Classical Unit Commitment formulation using MILP (OR-Tools).

Solves the unit commitment problem exactly:
  min  sum_t sum_i [ c_i * p_{i,t} + SU_i * startup_{i,t} ]
  s.t. sum_i p_{i,t} >= D_t                    (demand)
       P_min_i * x_{i,t} <= p_{i,t} <= P_max_i * x_{i,t}  (capacity)
       sum_i P_max_i * x_{i,t} >= D_t + reserve (reserve margin)

Variables:
  x_{i,t} in {0,1}  — generator i on/off at time t
  p_{i,t} in R       — power output of generator i at time t
  startup_{i,t} in {0,1} — 1 if generator i starts at time t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data.generators import GeneratorFleet


@dataclass
class UCResult:
    """Result of a unit commitment solve."""
    status: str                    # "optimal", "feasible", "infeasible", "error"
    total_cost: float = 0.0
    schedule: Optional[np.ndarray] = None    # (N, T) binary on/off
    dispatch: Optional[np.ndarray] = None    # (N, T) float power output MW
    solve_time_s: float = 0.0
    demand_met: bool = True
    constraint_violations: int = 0


def solve_unit_commitment(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    reserve_fraction: float = 0.10,
    time_limit_s: float = 30.0,
) -> UCResult:
    """Solve UC via MILP using OR-Tools CP-SAT or linear solver.

    Args:
        fleet: Generator fleet.
        demand: (T,) array of hourly demand in MW.
        reserve_fraction: Reserve margin as fraction of demand.
        time_limit_s: Solver time limit in seconds.

    Returns:
        UCResult with schedule, dispatch, and cost.
    """
    from ortools.linear_solver import pywraplp

    N = fleet.n_generators
    T = len(demand)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        # Fallback to GLOP won't work for MIP; try CBC
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return UCResult(status="error")

    solver.SetTimeLimit(int(time_limit_s * 1000))

    caps = fleet.capacity_vector()
    mins = fleet.min_output_vector()
    costs = fleet.cost_vector()
    su_costs = fleet.startup_cost_vector()

    # Decision variables
    x = {}   # binary on/off
    p = {}   # continuous power
    su = {}  # binary startup indicator

    for i in range(N):
        for t in range(T):
            x[i, t] = solver.BoolVar(f"x_{i}_{t}")
            p[i, t] = solver.NumVar(0, caps[i], f"p_{i}_{t}")
            su[i, t] = solver.BoolVar(f"su_{i}_{t}")

    # Constraints
    for t in range(T):
        # Demand satisfaction
        solver.Add(
            sum(p[i, t] for i in range(N)) >= demand[t],
            f"demand_{t}",
        )

        # Reserve margin
        solver.Add(
            sum(caps[i] * x[i, t] for i in range(N)) >= demand[t] * (1 + reserve_fraction),
            f"reserve_{t}",
        )

        for i in range(N):
            # Capacity bounds: min * x <= p <= max * x
            solver.Add(p[i, t] <= caps[i] * x[i, t], f"cap_upper_{i}_{t}")
            solver.Add(p[i, t] >= mins[i] * x[i, t], f"cap_lower_{i}_{t}")

            # Startup indicator: su_{i,t} >= x_{i,t} - x_{i,t-1}
            if t == 0:
                solver.Add(su[i, t] >= x[i, t], f"startup_{i}_{t}")
            else:
                solver.Add(su[i, t] >= x[i, t] - x[i, t - 1], f"startup_{i}_{t}")

    # Objective: minimize total fuel cost + startup cost
    objective = solver.Sum(
        costs[i] * p[i, t] + su_costs[i] * su[i, t]
        for i in range(N)
        for t in range(T)
    )
    solver.Minimize(objective)

    import time
    t0 = time.perf_counter()
    status_code = solver.Solve()
    solve_time = time.perf_counter() - t0

    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "error",
        pywraplp.Solver.NOT_SOLVED: "error",
    }
    status = status_map.get(status_code, "error")

    if status in ("optimal", "feasible"):
        schedule = np.zeros((N, T))
        dispatch = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                schedule[i, t] = x[i, t].solution_value()
                dispatch[i, t] = p[i, t].solution_value()

        total_cost = solver.Objective().Value()

        # Check demand satisfaction
        total_gen = dispatch.sum(axis=0)
        violations = int(np.sum(total_gen < demand - 1e-3))

        return UCResult(
            status=status,
            total_cost=total_cost,
            schedule=schedule,
            dispatch=dispatch,
            solve_time_s=solve_time,
            demand_met=(violations == 0),
            constraint_violations=violations,
        )

    return UCResult(status=status, solve_time_s=solve_time)


def compute_cost(
    fleet: GeneratorFleet,
    schedule: np.ndarray,
    dispatch: np.ndarray,
) -> float:
    """Compute total cost for a given schedule and dispatch.

    Args:
        fleet: Generator fleet.
        schedule: (N, T) binary on/off.
        dispatch: (N, T) power output MW.

    Returns:
        Total cost ($).
    """
    N, T = dispatch.shape
    costs = fleet.cost_vector()
    su_costs = fleet.startup_cost_vector()

    fuel = 0.0
    startup = 0.0
    for i in range(N):
        for t in range(T):
            fuel += costs[i] * dispatch[i, t]
            if t == 0:
                if schedule[i, t] > 0.5:
                    startup += su_costs[i]
            else:
                if schedule[i, t] > 0.5 and schedule[i, t - 1] < 0.5:
                    startup += su_costs[i]
    return fuel + startup


def check_feasibility(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    schedule: np.ndarray,
    dispatch: np.ndarray,
    reserve_fraction: float = 0.10,
) -> Dict[str, int]:
    """Check how many constraints are violated.

    Returns dict with violation counts per constraint type.
    """
    N, T = dispatch.shape
    caps = fleet.capacity_vector()
    mins = fleet.min_output_vector()

    violations = {
        "demand": 0,
        "reserve": 0,
        "capacity_upper": 0,
        "capacity_lower": 0,
    }

    for t in range(T):
        total_gen = sum(dispatch[i, t] for i in range(N))
        if total_gen < demand[t] - 1e-3:
            violations["demand"] += 1

        total_cap = sum(caps[i] * schedule[i, t] for i in range(N))
        if total_cap < demand[t] * (1 + reserve_fraction) - 1e-3:
            violations["reserve"] += 1

        for i in range(N):
            if dispatch[i, t] > caps[i] * schedule[i, t] + 1e-3:
                violations["capacity_upper"] += 1
            if dispatch[i, t] < mins[i] * schedule[i, t] - 1e-3:
                violations["capacity_lower"] += 1

    return violations
