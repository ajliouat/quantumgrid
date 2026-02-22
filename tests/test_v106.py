"""Tests for v1.0.6 — Penalty Tuning.

Covers:
  - Constraint checker correctness
  - Binary search convergence and result structure
  - Sensitivity sweep output
  - Fuel cost helper
"""
from __future__ import annotations

import numpy as np
import pytest

from data.generators import build_small_fleet
from formulation.qubo_encoding import uc_to_qubo, evaluate_qubo, decode_solution
from benchmarks.penalty_tuning import (
    TuningResult,
    SensitivityPoint,
    binary_search_lambda,
    check_constraints,
    sensitivity_sweep,
    _fuel_cost,
)


@pytest.fixture
def small_problem():
    fleet = build_small_fleet(n=4, seed=42)
    demand = np.array([300.0, 350.0, 400.0, 380.0, 320.0, 280.0])
    return fleet, demand


def _greedy_bitstring(fleet, demand, lam_d, lam_r):
    """Simple deterministic solution_fn for tuning tests: turn everyone on."""
    N = fleet.n_generators
    T = len(demand)
    return np.ones(N * T)


def _random_bitstring(fleet, demand, lam_d, lam_r):
    """Random solution for sweep tests."""
    rng = np.random.default_rng(int(lam_d * 100) % 2**31)
    return rng.integers(0, 2, size=fleet.n_generators * len(demand)).astype(float)


# ========== Constraint Checker ==========
class TestCheckConstraints:
    def test_all_on_meets_demand(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.ones((N, T))
        caps = fleet.capacity_vector()
        dispatch = schedule * caps[:, np.newaxis]  # full capacity
        # Clip dispatch to demand
        for t in range(T):
            total = dispatch[:, t].sum()
            if total > demand[t]:
                # Scale down proportionally
                dispatch[:, t] *= demand[t] / total
        check = check_constraints(fleet, demand, schedule, dispatch)
        assert check["demand_met_fraction"] >= 0.9

    def test_all_off_fails(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.zeros((N, T))
        dispatch = np.zeros((N, T))
        check = check_constraints(fleet, demand, schedule, dispatch)
        assert check["demand_met_fraction"] == 0.0
        assert np.all(check["demand_gap"] > 0)

    def test_return_keys(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.ones((N, T))
        dispatch = np.ones((N, T)) * 100
        check = check_constraints(fleet, demand, schedule, dispatch)
        for key in ["demand_met", "reserve_met", "demand_gap", "reserve_gap",
                     "demand_met_fraction", "reserve_met_fraction"]:
            assert key in check


# ========== Fuel Cost ==========
class TestFuelCost:
    def test_positive(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.ones((N, T))
        caps = fleet.capacity_vector()
        dispatch = schedule * caps[:, np.newaxis]
        cost = _fuel_cost(fleet, schedule, dispatch)
        assert cost > 0

    def test_zero_when_off(self, small_problem):
        fleet, demand = small_problem
        N, T = fleet.n_generators, len(demand)
        schedule = np.zeros((N, T))
        dispatch = np.zeros((N, T))
        cost = _fuel_cost(fleet, schedule, dispatch)
        assert cost == 0.0


# ========== Binary Search ==========
class TestBinarySearch:
    def test_result_type(self, small_problem):
        fleet, demand = small_problem
        res = binary_search_lambda(fleet, demand, _greedy_bitstring, max_steps=5)
        assert isinstance(res, TuningResult)

    def test_history_length(self, small_problem):
        fleet, demand = small_problem
        res = binary_search_lambda(fleet, demand, _greedy_bitstring, max_steps=5)
        assert len(res.search_history) == 5

    def test_optimal_lambda_positive(self, small_problem):
        fleet, demand = small_problem
        res = binary_search_lambda(fleet, demand, _greedy_bitstring, max_steps=5)
        assert res.optimal_lambda_demand > 0
        assert res.optimal_lambda_reserve > 0

    def test_satisfaction_recorded(self, small_problem):
        fleet, demand = small_problem
        res = binary_search_lambda(fleet, demand, _greedy_bitstring, max_steps=5)
        assert 0.0 <= res.constraint_satisfaction <= 1.0


# ========== Sensitivity Sweep ==========
class TestSensitivitySweep:
    def test_sweep_length(self, small_problem):
        fleet, demand = small_problem
        lambdas = np.array([10, 50, 100, 200, 500], dtype=float)
        pts = sensitivity_sweep(fleet, demand, _random_bitstring, lambdas)
        assert len(pts) == len(lambdas)

    def test_sweep_point_type(self, small_problem):
        fleet, demand = small_problem
        lambdas = np.array([50.0])
        pts = sensitivity_sweep(fleet, demand, _random_bitstring, lambdas)
        assert isinstance(pts[0], SensitivityPoint)
        assert pts[0].lambda_demand == 50.0

    def test_sweep_fractions_bounded(self, small_problem):
        fleet, demand = small_problem
        lambdas = np.array([10, 100, 1000], dtype=float)
        pts = sensitivity_sweep(fleet, demand, _random_bitstring, lambdas)
        for p in pts:
            assert 0.0 <= p.demand_met_fraction <= 1.0
            assert 0.0 <= p.reserve_met_fraction <= 1.0
