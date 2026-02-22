"""Visualisation utilities for QuantumGrid.

Provides matplotlib-based plotting functions:
  - Convergence curves (QAOA / VQE optimiser progress)
  - Scaling plots (solve time vs problem size)
  - Dispatch schedules (stacked area, generator timeline)
  - Sensitivity heatmaps
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    convergences: Dict[str, List[float]],
    title: str = "Optimiser Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot convergence curves for one or more solvers.

    Args:
        convergences: {label: [cost_at_each_iteration]}.
        title: Plot title.
        save_path: Optional file path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals in convergences.items():
        ax.plot(vals, label=label, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost (objective)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_scaling(
    records: List[Dict],
    x_key: str = "n_qubits",
    y_key: str = "time_s",
    group_key: str = "method",
    title: str = "Solve Time vs Problem Size",
    ylabel: str = "Time (s)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot scaling behaviour grouped by method.

    Args:
        records: List of dicts from ScalingStudy.to_records().
        x_key: Key for x-axis.
        y_key: Key for y-axis.
        group_key: Key to group (colour) by.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    groups: Dict[str, List] = {}
    for r in records:
        g = r[group_key]
        groups.setdefault(g, {"x": [], "y": []})
        groups[g]["x"].append(r[x_key])
        groups[g]["y"].append(r[y_key])

    for label, data in groups.items():
        ax.plot(data["x"], data["y"], "o-", label=label, linewidth=1.5, markersize=5)

    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_dispatch(
    dispatch: np.ndarray,
    demand: np.ndarray,
    generator_names: Optional[Sequence[str]] = None,
    title: str = "Generator Dispatch Schedule",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked area plot of generator dispatch vs demand.

    Args:
        dispatch: (N, T) dispatch matrix.
        demand: (T,) demand array.
        generator_names: Labels for generators.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    N, T = dispatch.shape
    if generator_names is None:
        generator_names = [f"Gen {i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(10, 5))
    timesteps = np.arange(T)

    ax.stackplot(timesteps, dispatch, labels=generator_names, alpha=0.7)
    ax.plot(timesteps, demand, "k--", linewidth=2, label="Demand")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_schedule_heatmap(
    schedule: np.ndarray,
    generator_names: Optional[Sequence[str]] = None,
    title: str = "Generator On/Off Schedule",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap showing generator on/off state over time.

    Args:
        schedule: (N, T) binary schedule.
        generator_names: Labels for generators.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    N, T = schedule.shape
    if generator_names is None:
        generator_names = [f"Gen {i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(8, T * 0.5), max(4, N * 0.5)))
    im = ax.imshow(schedule, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Generator")
    ax.set_yticks(range(N))
    ax.set_yticklabels(generator_names, fontsize=8)
    ax.set_xticks(range(T))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="On/Off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_sensitivity(
    lambda_values: np.ndarray,
    costs: np.ndarray,
    satisfaction: np.ndarray,
    title: str = "Penalty Sensitivity Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Dual-axis plot: cost and constraint satisfaction vs penalty weight.

    Args:
        lambda_values: (K,) penalty weights.
        costs: (K,) fuel costs.
        satisfaction: (K,) demand-met fractions.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(lambda_values, costs, "b-o", label="Fuel Cost", linewidth=1.5)
    ax2.plot(lambda_values, satisfaction, "r-s", label="Demand Met", linewidth=1.5)

    ax1.set_xlabel("Lambda (penalty weight)")
    ax1.set_ylabel("Fuel Cost", color="b")
    ax2.set_ylabel("Demand Met Fraction", color="r")
    ax1.set_title(title)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
