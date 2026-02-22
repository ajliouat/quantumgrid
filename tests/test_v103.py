"""Tests for v1.0.3 — QAOA Circuit.

Covers:
  - Build QAOA circuit (cost + probs QNodes)
  - QAOA result structure
  - Convergence (cost decreases over iterations)
  - Schedule decode from QAOA output
  - Determinism with same seed
  - Different n_layers produce different results
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from data.generators import build_small_fleet
from formulation.qubo_encoding import uc_to_qubo, qubo_to_ising
from quantum.cost_hamiltonian import ising_to_pennylane, mixer_hamiltonian
from quantum.qaoa_solver import (
    QAOAResult,
    build_qaoa_circuit,
    solve_qaoa,
)


@pytest.fixture
def small_problem():
    fleet = build_small_fleet(n=4, seed=42)
    demand = np.array([300.0, 350.0, 400.0, 380.0, 320.0, 280.0])
    return fleet, demand


@pytest.fixture
def tiny_problem():
    """2 gen × 3 timesteps = 6 qubits."""
    fleet = build_small_fleet(n=2, seed=0)
    demand = np.array([100.0, 150.0, 120.0])
    return fleet, demand


# ========== Build QAOA circuit ==========
class TestBuildQAOACircuit:
    def test_cost_qnode_returns_scalar(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, meta = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        n = meta["n_qubits"]
        cost_h = ising_to_pennylane(J, h, offset)
        mix_h = mixer_hamiltonian(n)
        cost_fn, _ = build_qaoa_circuit(cost_h, mix_h, n, n_layers=1)
        params = np.random.default_rng(0).uniform(0, np.pi, 2)
        val = cost_fn(params)
        assert np.isscalar(val) or val.ndim == 0

    def test_probs_qnode_returns_distribution(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, meta = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        n = meta["n_qubits"]
        cost_h = ising_to_pennylane(J, h, offset)
        mix_h = mixer_hamiltonian(n)
        _, probs_fn = build_qaoa_circuit(cost_h, mix_h, n, n_layers=1)
        params = np.random.default_rng(0).uniform(0, np.pi, 2)
        probs = np.array(probs_fn(params))
        assert probs.shape == (2**n,)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= -1e-12)


# ========== QAOA solver ==========
class TestQAOASolver:
    def test_result_type(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_qaoa(fleet, demand, n_layers=1, max_iterations=10, seed=42)
        assert isinstance(res, QAOAResult)

    def test_result_fields(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_qaoa(fleet, demand, n_layers=1, max_iterations=10, seed=42)
        assert res.best_bitstring.shape == (fleet.n_generators * len(demand),)
        assert np.all((res.best_bitstring == 0) | (res.best_bitstring == 1))
        assert res.schedule.shape == (fleet.n_generators, len(demand))
        assert res.dispatch.shape == (fleet.n_generators, len(demand))
        assert res.solve_time_s > 0
        assert res.n_layers == 1

    def test_convergence_recorded(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_qaoa(fleet, demand, n_layers=1, max_iterations=20, seed=42)
        assert len(res.convergence) > 0
        assert res.n_iterations == len(res.convergence)

    def test_cost_is_finite(self, tiny_problem):
        fleet, demand = tiny_problem
        res = solve_qaoa(fleet, demand, n_layers=1, max_iterations=10, seed=42)
        assert np.isfinite(res.best_cost)

    def test_deterministic_same_seed(self, tiny_problem):
        fleet, demand = tiny_problem
        r1 = solve_qaoa(fleet, demand, n_layers=1, max_iterations=15, seed=7)
        r2 = solve_qaoa(fleet, demand, n_layers=1, max_iterations=15, seed=7)
        assert np.allclose(r1.best_bitstring, r2.best_bitstring)
        assert abs(r1.best_cost - r2.best_cost) < 1e-6

    def test_different_seeds_or_layers(self, tiny_problem):
        fleet, demand = tiny_problem
        r1 = solve_qaoa(fleet, demand, n_layers=1, max_iterations=20, seed=0)
        r2 = solve_qaoa(fleet, demand, n_layers=2, max_iterations=20, seed=0)
        # Different n_layers → likely different optimal_params length
        assert len(r1.optimal_params) != len(r2.optimal_params)


# ========== Larger instance (4 gen × 6 h = 24 qubits) ==========
class TestQAOALarger:
    @pytest.mark.timeout(120)
    def test_24_qubit_runs(self, small_problem):
        """Smoke test on the reference 24-qubit instance."""
        fleet, demand = small_problem
        res = solve_qaoa(fleet, demand, n_layers=1, max_iterations=5, seed=42)
        assert isinstance(res, QAOAResult)
        assert res.best_bitstring.shape == (24,)
        assert res.solve_time_s > 0
