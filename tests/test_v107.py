"""Tests for v1.0.7 — Scaling Analysis.

Covers:
  - ScalingPoint and ScalingStudy data structures
  - Classical scaling (greedy + SA across problem sizes)
  - Quantum scaling (QAOA + VQE — tiny sizes only)
  - Filter and record export
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.scaling import (
    ScalingPoint,
    ScalingStudy,
    run_classical_scaling,
    run_quantum_scaling,
)


# ========== Data structures ==========
class TestScalingPoint:
    def test_creation(self):
        p = ScalingPoint(
            n_generators=4, n_timesteps=6, n_qubits=24,
            method="qaoa", layers=2,
            solve_time_s=1.5, total_cost=5000.0, demand_met_fraction=0.83,
        )
        assert p.n_qubits == 24
        assert p.method == "qaoa"


class TestScalingStudy:
    def test_to_records(self):
        s = ScalingStudy(points=[
            ScalingPoint(4, 6, 24, "qaoa", 1, 1.0, 5000, 0.8),
            ScalingPoint(4, 6, 24, "greedy", None, 0.01, 6000, 1.0),
        ])
        recs = s.to_records()
        assert len(recs) == 2
        assert recs[0]["method"] == "qaoa"

    def test_filter_method(self):
        s = ScalingStudy(points=[
            ScalingPoint(4, 6, 24, "qaoa", 1, 1.0, 5000, 0.8),
            ScalingPoint(4, 6, 24, "greedy", None, 0.01, 6000, 1.0),
            ScalingPoint(4, 6, 24, "qaoa", 2, 2.0, 4500, 0.9),
        ])
        qaoa_pts = s.filter_method("qaoa")
        assert len(qaoa_pts) == 2


# ========== Classical scaling ==========
class TestClassicalScaling:
    def test_runs(self):
        study = run_classical_scaling(
            n_gen_list=[4], t_list=[6], sa_iterations=100
        )
        assert len(study.points) == 2  # greedy + SA

    def test_all_positive_costs(self):
        study = run_classical_scaling(
            n_gen_list=[4, 6], t_list=[6], sa_iterations=100
        )
        for p in study.points:
            assert p.total_cost > 0
            assert p.solve_time_s >= 0

    def test_multiple_sizes(self):
        study = run_classical_scaling(
            n_gen_list=[4, 6], t_list=[6, 12], sa_iterations=50
        )
        # 2 sizes × 2 horizons × 2 methods = 8
        assert len(study.points) == 8


# ========== Quantum scaling (tiny) ==========
class TestQuantumScaling:
    @pytest.mark.timeout(120)
    def test_runs_tiny(self):
        """2 generators × 3 timesteps = 6 qubits — fast enough to test."""
        study = run_quantum_scaling(
            n_gen_list=[2], t_list=[3], p_list=[1],
            max_qaoa_iter=5, max_vqe_iter=5,
        )
        # 1 size × 1 horizon × (1 QAOA layer + 1 VQE) = 2
        assert len(study.points) == 2
        for p in study.points:
            assert p.solve_time_s > 0
            assert np.isfinite(p.total_cost)

    @pytest.mark.timeout(120)
    def test_methods_present(self):
        study = run_quantum_scaling(
            n_gen_list=[2], t_list=[3], p_list=[1],
            max_qaoa_iter=5, max_vqe_iter=5,
        )
        methods = {p.method for p in study.points}
        assert "qaoa" in methods
        assert "vqe" in methods
