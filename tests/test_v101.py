"""v1.0.1 tests — Classical MILP Solver.

Covers:
  - UC formulation correctness
  - MILP solver on small instances (4-6 generators, 6 hours)
  - Solution feasibility checks
  - Cost computation
  - Solver wrapper interface
"""
from __future__ import annotations

import numpy as np
import pytest

from data.generators import build_small_fleet, build_fleet
from data.download_entsoe import fetch_load_forecast
from data.preprocess import extract_horizon
from formulation.unit_commitment import (
    UCResult,
    solve_unit_commitment,
    compute_cost,
    check_feasibility,
)
from classical.milp_solver import solve_milp


@pytest.fixture
def small_fleet():
    return build_small_fleet(n=4, seed=42)


@pytest.fixture
def demand_6h(small_fleet):
    """6-hour demand that the small fleet can cover."""
    total_cap = small_fleet.total_capacity
    # Demand at ~50-70% of total capacity
    rng = np.random.default_rng(42)
    return total_cap * (0.5 + 0.2 * rng.random(6))


class TestUCResult:
    def test_dataclass_defaults(self):
        r = UCResult(status="optimal")
        assert r.total_cost == 0.0
        assert r.schedule is None
        assert r.demand_met is True

    def test_dataclass_fields(self):
        r = UCResult(
            status="optimal",
            total_cost=1234.5,
            schedule=np.ones((4, 6)),
            dispatch=np.ones((4, 6)) * 100,
            solve_time_s=0.5,
        )
        assert r.status == "optimal"
        assert r.total_cost == 1234.5
        assert r.schedule.shape == (4, 6)


class TestMILPSolver:
    def test_solve_small_instance(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        assert result.status in ("optimal", "feasible")
        assert result.total_cost > 0
        assert result.schedule is not None
        assert result.dispatch is not None

    def test_solution_shape(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        assert result.schedule.shape == (4, 6)
        assert result.dispatch.shape == (4, 6)

    def test_schedule_binary(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        # Schedule should be 0 or 1 (within tolerance)
        for val in result.schedule.flat:
            assert val < 0.1 or val > 0.9

    def test_demand_satisfied(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        total_gen = result.dispatch.sum(axis=0)
        for t in range(6):
            assert total_gen[t] >= demand_6h[t] - 1e-3

    def test_capacity_respected(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        caps = small_fleet.capacity_vector()
        for i in range(4):
            for t in range(6):
                assert result.dispatch[i, t] <= caps[i] + 1e-3

    def test_off_generators_zero_output(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        for i in range(4):
            for t in range(6):
                if result.schedule[i, t] < 0.5:
                    assert result.dispatch[i, t] < 1e-3

    def test_solve_time_reasonable(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        assert result.solve_time_s < 30.0  # should be < 1s for small

    def test_six_generator_instance(self):
        fleet = build_small_fleet(n=6, seed=11)
        total_cap = fleet.total_capacity
        demand = total_cap * np.full(6, 0.5)
        result = solve_unit_commitment(fleet, demand)
        assert result.status in ("optimal", "feasible")


class TestCostComputation:
    def test_compute_cost_basic(self, small_fleet):
        schedule = np.ones((4, 6))   # all on
        dispatch = np.ones((4, 6)) * 100  # 100 MW each
        cost = compute_cost(small_fleet, schedule, dispatch)
        assert cost > 0

    def test_compute_cost_zero_dispatch(self, small_fleet):
        schedule = np.zeros((4, 6))
        dispatch = np.zeros((4, 6))
        cost = compute_cost(small_fleet, schedule, dispatch)
        assert cost == 0.0

    def test_cost_matches_solver(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        if result.status in ("optimal", "feasible"):
            recomputed = compute_cost(small_fleet, result.schedule, result.dispatch)
            # Should be close (startup detection may differ slightly)
            assert abs(recomputed - result.total_cost) / max(result.total_cost, 1) < 0.1


class TestFeasibilityCheck:
    def test_feasible_solution(self, small_fleet, demand_6h):
        result = solve_unit_commitment(small_fleet, demand_6h)
        violations = check_feasibility(
            small_fleet, demand_6h, result.schedule, result.dispatch,
        )
        assert violations["demand"] == 0
        assert violations["capacity_upper"] == 0

    def test_infeasible_detection(self, small_fleet):
        # Demand way too high — no solution can satisfy it
        huge_demand = np.full(6, 1e9)
        schedule = np.ones((4, 6))
        dispatch = np.ones((4, 6)) * 100
        violations = check_feasibility(
            small_fleet, huge_demand, schedule, dispatch,
        )
        assert violations["demand"] > 0


class TestMILPWrapper:
    def test_wrapper_interface(self, small_fleet, demand_6h):
        result = solve_milp(small_fleet, demand_6h)
        assert isinstance(result, UCResult)
        assert result.status in ("optimal", "feasible")
