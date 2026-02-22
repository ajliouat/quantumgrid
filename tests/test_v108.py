"""Tests for v1.0.8 — Visualisation.

Covers:
  - All plot functions return valid Figure objects
  - Save-to-file works
  - Edge cases (single generator, single timestep)
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualization.plots import (
    plot_convergence,
    plot_dispatch,
    plot_scaling,
    plot_schedule_heatmap,
    plot_sensitivity,
)


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


# ========== Convergence ==========
class TestConvergence:
    def test_returns_figure(self):
        fig = plot_convergence({"QAOA": [10, 8, 6, 5, 5]})
        assert isinstance(fig, plt.Figure)

    def test_multiple_curves(self):
        fig = plot_convergence({
            "QAOA p=1": [10, 8, 6],
            "QAOA p=2": [9, 6, 4],
            "VQE": [12, 7, 5],
        })
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_convergence({"test": [5, 4, 3]}, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ========== Scaling ==========
class TestScaling:
    def test_returns_figure(self):
        records = [
            {"n_qubits": 6, "time_s": 0.5, "method": "qaoa"},
            {"n_qubits": 12, "time_s": 2.0, "method": "qaoa"},
            {"n_qubits": 6, "time_s": 0.01, "method": "greedy"},
            {"n_qubits": 12, "time_s": 0.02, "method": "greedy"},
        ]
        fig = plot_scaling(records)
        assert isinstance(fig, plt.Figure)

    def test_custom_keys(self):
        records = [
            {"n_gen": 4, "cost": 5000, "method": "qaoa"},
            {"n_gen": 6, "cost": 7000, "method": "qaoa"},
        ]
        fig = plot_scaling(records, x_key="n_gen", y_key="cost", ylabel="Cost")
        assert isinstance(fig, plt.Figure)


# ========== Dispatch ==========
class TestDispatch:
    def test_returns_figure(self):
        dispatch = np.array([[100, 200, 150], [50, 100, 80]])
        demand = np.array([140, 280, 220])
        fig = plot_dispatch(dispatch, demand)
        assert isinstance(fig, plt.Figure)

    def test_with_names(self):
        dispatch = np.array([[100, 200], [50, 100]])
        demand = np.array([140, 280])
        fig = plot_dispatch(dispatch, demand, generator_names=["Nuclear", "Gas"])
        assert isinstance(fig, plt.Figure)

    def test_single_gen(self):
        dispatch = np.array([[100, 200, 150]])
        demand = np.array([90, 180, 140])
        fig = plot_dispatch(dispatch, demand)
        assert isinstance(fig, plt.Figure)


# ========== Schedule heatmap ==========
class TestHeatmap:
    def test_returns_figure(self):
        schedule = np.array([[1, 1, 0], [0, 1, 1]])
        fig = plot_schedule_heatmap(schedule)
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            schedule = np.array([[1, 0], [1, 1]])
            plot_schedule_heatmap(schedule, save_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ========== Sensitivity ==========
class TestSensitivity:
    def test_returns_figure(self):
        lambdas = np.array([10, 50, 100, 500, 1000], dtype=float)
        costs = np.array([3000, 4000, 5000, 5500, 5500], dtype=float)
        sat = np.array([0.3, 0.6, 0.8, 0.95, 1.0])
        fig = plot_sensitivity(lambdas, costs, sat)
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            lambdas = np.array([10, 100], dtype=float)
            costs = np.array([3000, 5000], dtype=float)
            sat = np.array([0.5, 1.0])
            plot_sensitivity(lambdas, costs, sat, save_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)
