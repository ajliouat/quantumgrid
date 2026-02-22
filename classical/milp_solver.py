"""Classical MILP solver wrapper for unit commitment.

Thin wrapper around formulation.unit_commitment that provides a unified
solver interface consistent with quantum and heuristic solvers.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from data.generators import GeneratorFleet
from formulation.unit_commitment import (
    UCResult,
    solve_unit_commitment,
    compute_cost,
    check_feasibility,
)


def solve_milp(
    fleet: GeneratorFleet,
    demand: np.ndarray,
    reserve_fraction: float = 0.10,
    time_limit_s: float = 30.0,
) -> UCResult:
    """Solve UC with MILP. Unified interface for benchmarks."""
    return solve_unit_commitment(
        fleet=fleet,
        demand=demand,
        reserve_fraction=reserve_fraction,
        time_limit_s=time_limit_s,
    )
