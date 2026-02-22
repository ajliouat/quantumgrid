"""Tests for v1.0.2 — QUBO Encoding & Ising Hamiltonian.

Covers:
  - QUBO matrix construction from UC problem
  - QUBO → Ising conversion identity
  - Evaluate QUBO / Ising consistency
  - Decode solution back to schedule/dispatch
  - PennyLane Hamiltonian construction
  - Mixer Hamiltonian structure
  - Bitstring energy evaluation
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from data.generators import build_small_fleet
from formulation.qubo_encoding import (
    decode_solution,
    evaluate_ising,
    evaluate_qubo,
    qubo_to_ising,
    uc_to_qubo,
)
from quantum.cost_hamiltonian import (
    bitstring_energy,
    ising_to_pennylane,
    mixer_hamiltonian,
    num_hamiltonian_terms,
)


# ---------- fixtures ----------
@pytest.fixture
def small_problem():
    """4 generators, 6 timesteps — the reference quantum-scale instance."""
    fleet = build_small_fleet(n=4, seed=42)
    demand = np.array([300.0, 350.0, 400.0, 380.0, 320.0, 280.0])
    return fleet, demand


@pytest.fixture
def tiny_problem():
    """2 generators, 3 timesteps — ultra-small for exact checks."""
    fleet = build_small_fleet(n=2, seed=0)
    demand = np.array([100.0, 150.0, 120.0])
    return fleet, demand


# ========== QUBO construction ==========
class TestQUBOConstruction:
    def test_qubo_shape(self, small_problem):
        fleet, demand = small_problem
        Q, meta = uc_to_qubo(fleet, demand)
        n = fleet.n_generators * len(demand)
        assert Q.shape == (n, n)

    def test_qubo_metadata(self, small_problem):
        fleet, demand = small_problem
        Q, meta = uc_to_qubo(fleet, demand)
        assert meta["n_generators"] == 4
        assert meta["n_timesteps"] == 6
        assert meta["n_qubits"] == 24
        assert len(meta["variable_map"]) == 24

    def test_qubo_not_all_zero(self, small_problem):
        fleet, demand = small_problem
        Q, _ = uc_to_qubo(fleet, demand)
        assert np.any(Q != 0)

    def test_qubo_upper_triangle_has_entries(self, small_problem):
        fleet, demand = small_problem
        Q, _ = uc_to_qubo(fleet, demand)
        upper = np.triu(Q, k=1)
        assert np.any(upper != 0), "Expect off-diagonal coupling terms"

    def test_qubo_different_penalties(self, small_problem):
        fleet, demand = small_problem
        Q1, _ = uc_to_qubo(fleet, demand, lambda_demand=10)
        Q2, _ = uc_to_qubo(fleet, demand, lambda_demand=200)
        assert not np.allclose(Q1, Q2)


# ========== QUBO ↔ Ising conversion ==========
class TestQUBOIsing:
    def test_ising_shapes(self, small_problem):
        fleet, demand = small_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        n = Q.shape[0]
        assert J.shape == (n, n)
        assert h.shape == (n,)

    def test_ising_upper_triangular(self, small_problem):
        fleet, demand = small_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, _, _ = qubo_to_ising(Q)
        assert np.allclose(J, np.triu(J)), "J should be upper triangular"

    def test_qubo_ising_energy_equivalence(self, tiny_problem):
        """For every bitstring, QUBO(x) == Ising(z) with z = 1-2x."""
        fleet, demand = tiny_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        n = Q.shape[0]
        for bits in range(2**n):
            x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            z = 1 - 2 * x
            e_qubo = evaluate_qubo(Q, x)
            e_ising = evaluate_ising(J, h, offset, z)
            assert abs(e_qubo - e_ising) < 1e-6, (
                f"Mismatch at bits={bits:0{n}b}: QUBO={e_qubo}, Ising={e_ising}"
            )

    def test_all_off_cost_zero_qubo(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, _ = uc_to_qubo(fleet, demand)
        n = Q.shape[0]
        x = np.zeros(n)
        # All off => no generation cost, but huge demand violation penalty
        # The point is evaluate_qubo should be deterministic
        e = evaluate_qubo(Q, x)
        assert np.isfinite(e)


# ========== Decode solution ==========
class TestDecodeSolution:
    def test_decode_shape(self, small_problem):
        fleet, demand = small_problem
        n_qubits = fleet.n_generators * len(demand)
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=n_qubits).astype(float)
        schedule, dispatch = decode_solution(x, fleet, len(demand))
        assert schedule.shape == (fleet.n_generators, len(demand))
        assert dispatch.shape == (fleet.n_generators, len(demand))

    def test_decode_dispatch_respects_schedule(self, small_problem):
        fleet, demand = small_problem
        n_qubits = fleet.n_generators * len(demand)
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=n_qubits).astype(float)
        schedule, dispatch = decode_solution(x, fleet, len(demand))
        # Where off, dispatch is 0
        off_mask = schedule == 0
        assert np.all(dispatch[off_mask] == 0)

    def test_decode_all_on(self, small_problem):
        fleet, demand = small_problem
        T = len(demand)
        x = np.ones(fleet.n_generators * T)
        schedule, dispatch = decode_solution(x, fleet, T)
        caps = fleet.capacity_vector()
        for t in range(T):
            assert np.allclose(dispatch[:, t], caps)


# ========== PennyLane Hamiltonian ==========
class TestPennyLaneHamiltonian:
    def test_hamiltonian_creation(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        H = ising_to_pennylane(J, h, offset)
        assert isinstance(H, qml.Hamiltonian)
        assert len(H.coeffs) > 0

    def test_hamiltonian_n_qubits(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, meta = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        H = ising_to_pennylane(J, h, offset)
        # Matrix should be 2^n x 2^n
        mat = qml.matrix(H)
        n = meta["n_qubits"]
        assert mat.shape == (2**n, 2**n)

    def test_hamiltonian_hermitian(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        H = ising_to_pennylane(J, h, offset)
        mat = qml.matrix(H)
        assert np.allclose(mat, mat.conj().T), "Hamiltonian must be Hermitian"

    def test_bitstring_energy_matches_ising(self, tiny_problem):
        """bitstring_energy via PennyLane matrix == evaluate_ising."""
        fleet, demand = tiny_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        H = ising_to_pennylane(J, h, offset)
        n = Q.shape[0]
        for bits in range(2**n):
            x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            z = 1 - 2 * x
            e_pl = bitstring_energy(H, x)
            e_ising = evaluate_ising(J, h, offset, z)
            assert abs(e_pl - e_ising) < 1e-4, (
                f"Mismatch at bits={bits:0{n}b}: PL={e_pl}, Ising={e_ising}"
            )


# ========== Mixer Hamiltonian ==========
class TestMixerHamiltonian:
    def test_mixer_n_terms(self):
        H = mixer_hamiltonian(6)
        assert num_hamiltonian_terms(H) == 6

    def test_mixer_hermitian(self):
        H = mixer_hamiltonian(4)
        mat = qml.matrix(H)
        assert np.allclose(mat, mat.conj().T)

    def test_mixer_eigenvalues(self):
        """X-mixer on 2 qubits: eigenvalues should be {-2, 0, 0, 2}."""
        H = mixer_hamiltonian(2)
        mat = qml.matrix(H)
        evals = np.sort(np.linalg.eigvalsh(mat))
        expected = np.array([-2, 0, 0, 2], dtype=float)
        assert np.allclose(evals, expected)


# ========== Num terms ==========
class TestNumTerms:
    def test_cost_hamiltonian_terms(self, tiny_problem):
        fleet, demand = tiny_problem
        Q, _ = uc_to_qubo(fleet, demand)
        J, h, offset = qubo_to_ising(Q)
        H = ising_to_pennylane(J, h, offset)
        nt = num_hamiltonian_terms(H)
        assert nt > 0
        # Upper bound: 1 (offset) + n (linear) + n*(n-1)/2 (quadratic)
        n = Q.shape[0]
        max_terms = 1 + n + n * (n - 1) // 2
        assert nt <= max_terms
