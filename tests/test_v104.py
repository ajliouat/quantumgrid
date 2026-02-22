"""Tests for v1.0.4 — VQE Solver.

Covers:
  - Hardware-efficient ansatz circuit construction
  - VQE result structure and fields
  - Convergence recording
  - Determinism with fixed seed
  - Different ansatz depths
  - 24-qubit smoke test
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from data.generators import build_small_fleet
from formulation.qubo_encoding import uc_to_qubo, qubo_to_ising
from quantum.cost_hamiltonian import ising_to_pennylane
from quantum.vqe_solver import (
    VQEResult,
    build_vqe_circuit,
    solve_vqe,
)


@pytest.fixture
def tiny_problem():
    fleet = build_small_fleet(n=2, seed=0)
    demand = np.array([100.0, 150.0, 120.0])
    return fleet, demand


@pytest.fixture
def small_problem():
    fleet = build_small_fleet(n=4, seed=42)
    demand = np.array([300.0, 350.0, 400.0, 380.0, 320.0, 280.0])
    return fleet, demand


# ========== Circuit construction ==========
class TestVQECircuit:
    def test_cost_returns_scalar(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, meta = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        n = meta["n_qubits"]
        cost_h = ising_to_pennylane(J, h, offset)
        cost_fn, _ = build_vqe_circuit(cost_h, n, n_layers=1)
        params = np.random.default_rng(0).uniform(-np.pi, np.pi, 1 * n * 2)
        val = cost_fn(params)
        assert np.isscalar(val) or val.ndim == 0

    def test_probs_valid_distribution(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, meta = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        n = meta["n_qubits"]
        cost_h = ising_to_pennylane(J, h, offset)
        _, probs_fn = build_vqe_circuit(cost_h, n, n_layers=1)
        params = np.random.default_rng(0).uniform(-np.pi, np.pi, 1 * n * 2)
        probs = np.array(probs_fn(params))
        assert probs.shape == (2**n,)
        assert abs(probs.sum() - 1.0) < 1e-6


# ========== VQE solver ==========
class TestVQESolver:
    def test_result_type(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_vqe(fleet, demand, n_layers=1, max_iterations=10, seed=42)
        assert isinstance(res, VQEResult)

    def test_result_fields(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_vqe(fleet, demand, n_layers=1, max_iterations=10, seed=42)
        n = fleet.n_generators * len(demand)
        assert res.best_bitstring.shape == (n,)
        assert np.all((res.best_bitstring == 0) | (res.best_bitstring == 1))
        assert res.schedule.shape == (fleet.n_generators, len(demand))
        assert res.dispatch.shape == (fleet.n_generators, len(demand))
        assert res.solve_time_s > 0
        assert res.ansatz == "hardware_efficient_ry_rz"

    def test_convergence_recorded(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_vqe(fleet, demand, n_layers=1, max_iterations=20, seed=42)
        assert len(res.convergence) > 0
        assert res.n_iterations == len(res.convergence)

    def test_cost_is_finite(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_vqe(fleet, demand, n_layers=1, max_iterations=10, seed=42)
        assert np.isfinite(res.best_cost)

    def test_deterministic_same_seed(self, tiny_problem):
        fleet, demand = tiny_problem
        r1 = solve_vqe(fleet, demand, n_layers=1, max_iterations=15, seed=7)
        r2 = solve_vqe(fleet, demand, n_layers=1, max_iterations=15, seed=7)
        assert np.allclose(r1.best_bitstring, r2.best_bitstring)
        assert abs(r1.best_cost - r2.best_cost) < 1e-6

    def test_different_layers_different_params(self, tiny_problem):
        fleet, demand = tiny_problem
        r1 = solve_vqe(fleet, demand, n_layers=1, max_iterations=10, seed=0)
        r2 = solve_vqe(fleet, demand, n_layers=2, max_iterations=10, seed=0)
        assert len(r1.optimal_params) != len(r2.optimal_params)


# ========== Medium instance smoke test (3 gen × 4 h = 12 qubits) ==========
class TestVQEMedium:
    @pytest.mark.timeout(120)
    def test_12_qubit_runs(self):
        fleet = build_small_fleet(n=3, seed=42)
        demand = np.array([200.0, 250.0, 220.0, 180.0])
        res = solve_vqe(fleet, demand, n_layers=1, max_iterations=5, seed=42)
        assert isinstance(res, VQEResult)
        assert res.best_bitstring.shape == (12,)
        assert res.solve_time_s > 0
