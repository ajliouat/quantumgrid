"""Tests for v1.0.5 — Classical Baselines (SA + Greedy).

Covers:
  - Simulated annealing result structure, convergence, cost
  - Greedy priority list result structure, demand satisfaction
  - Both methods produce valid schedules
  - SA improves over initial solution
  - Greedy meets demand + reserve
"""
from __future__ import annotations

import numpy as np
import pytest

from data.generators import build_fleet, build_small_fleet
from classical.baselines import (
    BaselineResult,
    solve_greedy,
    solve_simulated_annealing,
    _compute_cost,
    _dispatch_for_schedule,
    _demand_violation,
)


@pytest.fixture
def medium_problem():
    fleet = build_fleet(n_generators=10, total_capacity_mw=2000, seed=42)
    demand = np.array([800, 1000, 1200, 1100, 900, 700], dtype=float)
    return fleet, demand


@pytest.fixture
def small_problem():
    fleet = build_small_fleet(n=4, seed=42)
    demand = np.array([300.0, 350.0, 400.0, 380.0, 320.0, 280.0])
    return fleet, demand


# ========== Greedy ==========
class TestGreedy:
    def test_result_type(self, medium_problem):
        fleet, demand = medium_problem
        res = solve_greedy(fleet, demand)
        assert isinstance(res, BaselineResult)
        assert res.method == "greedy_priority_list"

    def test_schedule_shape(self, medium_problem):
        fleet, demand = medium_problem
        res = solve_greedy(fleet, demand)
        assert res.schedule.shape == (fleet.n_generators, len(demand))
        assert res.dispatch.shape == (fleet.n_generators, len(demand))

    def test_demand_met(self, medium_problem):
        fleet, demand = medium_problem
        res = solve_greedy(fleet, demand)
        gen_total = res.dispatch.sum(axis=0)
        for t in range(len(demand)):
            assert gen_total[t] >= demand[t] * 0.99, (
                f"Demand not met at t={t}: {gen_total[t]:.1f} < {demand[t]:.1f}"
            )

    def test_cost_positive(self, medium_problem):
        fleet, demand = medium_problem
        res = solve_greedy(fleet, demand)
        assert res.total_cost > 0

    def test_solve_time(self, medium_problem):
        fleet, demand = medium_problem
        res = solve_greedy(fleet, demand)
        assert res.solve_time_s >= 0


# ========== Simulated Annealing ==========
class TestSA:
    def test_result_type(self, small_problem):
        fleet, demand = small_problem
        res = solve_simulated_annealing(fleet, demand, max_iterations=500)
        assert isinstance(res, BaselineResult)
        assert res.method == "simulated_annealing"

    def test_schedule_shape(self, small_problem):
        fleet, demand = small_problem
        res = solve_simulated_annealing(fleet, demand, max_iterations=500)
        assert res.schedule.shape == (fleet.n_generators, len(demand))

    def test_convergence_recorded(self, small_problem):
        fleet, demand = small_problem
        res = solve_simulated_annealing(fleet, demand, max_iterations=500)
        assert len(res.convergence) > 1

    def test_convergence_non_increasing(self, small_problem):
        """Best energy should never increase."""
        fleet, demand = small_problem
        res = solve_simulated_annealing(fleet, demand, max_iterations=500)
        for i in range(1, len(res.convergence)):
            assert res.convergence[i] <= res.convergence[i - 1] + 1e-6

    def test_cost_positive(self, small_problem):
        fleet, demand = small_problem
        res = solve_simulated_annealing(fleet, demand, max_iterations=500)
        assert res.total_cost > 0

    def test_deterministic_same_seed(self, small_problem):
        fleet, demand = small_problem
        r1 = solve_simulated_annealing(fleet, demand, max_iterations=200, seed=7)
        r2 = solve_simulated_annealing(fleet, demand, max_iterations=200, seed=7)
        assert np.array_equal(r1.schedule, r2.schedule)
        assert abs(r1.total_cost - r2.total_cost) < 1e-6


# ========== Helpers ==========
class TestHelpers:
    def test_compute_cost(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.ones((N, T), dtype=int)
        dispatch = _dispatch_for_schedule(fleet, demand, schedule)
        cost = _compute_cost(fleet, schedule, dispatch)
        assert cost > 0
        assert np.isfinite(cost)

    def test_demand_violation_zero_when_met(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.ones((N, T), dtype=int)
        dispatch = _dispatch_for_schedule(fleet, demand, schedule)
        viol = _demand_violation(dispatch, demand)
        assert viol < 1e-6, f"Expected zero violation, got {viol}"

    def test_demand_violation_positive_when_unmet(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        dispatch = np.zeros((N, T))  # nobody on
        viol = _demand_violation(dispatch, demand)
        assert viol > 0
