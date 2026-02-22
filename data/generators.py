"""Generator fleet model for unit commitment.

Constructs a synthetic fleet calibrated to match a country's aggregate
installed capacity. Each generator has:
  - name, fuel_type
  - capacity_mw (max output)
  - min_output_mw (minimum stable generation when on)
  - fuel_cost ($/MWh marginal cost)
  - startup_cost ($)
  - min_up_time (hours)
  - min_down_time (hours)
  - ramp_rate (MW/hour)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Generator:
    """Single power generator."""
    name: str
    fuel_type: str
    capacity_mw: float
    min_output_mw: float
    fuel_cost: float          # $/MWh
    startup_cost: float       # $
    min_up_time: int = 1      # hours
    min_down_time: int = 1    # hours
    ramp_rate: float = 1e9    # MW/h (default: unlimited)


@dataclass
class GeneratorFleet:
    """Collection of generators for unit commitment."""
    generators: List[Generator] = field(default_factory=list)

    @property
    def n_generators(self) -> int:
        return len(self.generators)

    @property
    def total_capacity(self) -> float:
        return sum(g.capacity_mw for g in self.generators)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fleet to DataFrame."""
        records = []
        for g in self.generators:
            records.append({
                "name": g.name,
                "fuel_type": g.fuel_type,
                "capacity_mw": g.capacity_mw,
                "min_output_mw": g.min_output_mw,
                "fuel_cost": g.fuel_cost,
                "startup_cost": g.startup_cost,
                "min_up_time": g.min_up_time,
                "min_down_time": g.min_down_time,
                "ramp_rate": g.ramp_rate,
            })
        return pd.DataFrame(records)

    def cost_vector(self) -> np.ndarray:
        """Marginal cost vector, shape (N,)."""
        return np.array([g.fuel_cost for g in self.generators])

    def capacity_vector(self) -> np.ndarray:
        """Max capacity vector, shape (N,)."""
        return np.array([g.capacity_mw for g in self.generators])

    def min_output_vector(self) -> np.ndarray:
        """Min output vector, shape (N,)."""
        return np.array([g.min_output_mw for g in self.generators])

    def startup_cost_vector(self) -> np.ndarray:
        """Startup cost vector, shape (N,)."""
        return np.array([g.startup_cost for g in self.generators])


def build_fleet(
    n_generators: int = 10,
    total_capacity_mw: float = 60000.0,
    seed: int = 42,
) -> GeneratorFleet:
    """Build a synthetic generator fleet.

    Creates a mix of nuclear, gas, coal, and peaker generators calibrated
    so total capacity approximately equals total_capacity_mw.
    """
    rng = np.random.default_rng(seed)

    # Fuel type templates: (type, frac_of_fleet, cost_range, startup_range,
    #                       min_up, min_down, capacity_frac_of_unit, min_output_frac)
    templates = [
        ("nuclear", 0.35, (8, 12), (50000, 80000), 24, 24, (0.8, 1.2), 0.5),
        ("gas_ccgt", 0.25, (35, 55), (10000, 20000), 4, 4, (0.6, 1.0), 0.3),
        ("coal", 0.15, (25, 40), (20000, 40000), 8, 8, (0.5, 0.9), 0.4),
        ("gas_peaker", 0.15, (60, 90), (5000, 10000), 1, 1, (0.3, 0.6), 0.2),
        ("hydro", 0.10, (3, 8), (1000, 3000), 1, 1, (0.4, 0.8), 0.1),
    ]

    generators = []
    gen_id = 0

    for fuel_type, frac, cost_range, su_range, mu, md, cap_frac, min_frac in templates:
        n_of_type = max(1, round(frac * n_generators))
        cap_per_unit = (total_capacity_mw * frac) / n_of_type

        for i in range(n_of_type):
            if gen_id >= n_generators:
                break
            scale = rng.uniform(*cap_frac)
            cap = cap_per_unit * scale
            generators.append(Generator(
                name=f"{fuel_type}_{i}",
                fuel_type=fuel_type,
                capacity_mw=round(cap, 1),
                min_output_mw=round(cap * min_frac, 1),
                fuel_cost=round(rng.uniform(*cost_range), 2),
                startup_cost=round(rng.uniform(*su_range), 0),
                min_up_time=mu,
                min_down_time=md,
                ramp_rate=round(cap * 0.3, 1),  # 30% of capacity per hour
            ))
            gen_id += 1

    # Trim or pad to exactly n_generators
    generators = generators[:n_generators]

    return GeneratorFleet(generators=generators)


def build_small_fleet(n: int = 4, seed: int = 42) -> GeneratorFleet:
    """Build a small fleet for quantum solver testing.

    All generators are simple: no min up/down constraints, unlimited ramp.
    """
    rng = np.random.default_rng(seed)
    generators = []
    fuel_types = ["nuclear", "gas", "coal", "peaker", "hydro", "wind"]

    for i in range(n):
        ft = fuel_types[i % len(fuel_types)]
        cap = round(rng.uniform(100, 500), 1)
        generators.append(Generator(
            name=f"gen_{i}",
            fuel_type=ft,
            capacity_mw=cap,
            min_output_mw=round(cap * 0.2, 1),
            fuel_cost=round(rng.uniform(10, 80), 2),
            startup_cost=round(rng.uniform(500, 5000), 0),
            min_up_time=1,
            min_down_time=1,
            ramp_rate=cap,
        ))

    return GeneratorFleet(generators=generators)
