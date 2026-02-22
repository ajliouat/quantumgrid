"""v1.0.0 tests — Scaffold, data ingestion, generator fleet.

Covers:
  - ENTSO-E download (synthetic fallback)
  - Load profile shapes and value ranges
  - Generation mix shapes and fuel types
  - Preprocessing: normalization, resampling, horizon extraction, train/test
  - Generator fleet: construction, properties, small fleet
  - Data consistency
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.download_entsoe import (
    ZONES,
    DOC_TYPES,
    fetch_load_forecast,
    fetch_generation_mix,
)
from data.preprocess import (
    normalize_load,
    resample_hourly,
    extract_horizon,
    train_test_split_temporal,
)
from data.generators import (
    Generator,
    GeneratorFleet,
    build_fleet,
    build_small_fleet,
)


# ── ENTSO-E Download / Synthetic ──────────────────────────────────────

class TestDownload:
    def test_zones_defined(self):
        assert "FR" in ZONES
        assert "DE" in ZONES

    def test_doc_types_defined(self):
        assert "day_ahead_load" in DOC_TYPES
        assert "actual_generation" in DOC_TYPES

    def test_load_forecast_shape(self):
        df = fetch_load_forecast(zone="FR", year=2023)
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "load_mw" in df.columns
        assert len(df) == 8760

    def test_load_forecast_values(self):
        df = fetch_load_forecast()
        assert df["load_mw"].min() >= 20000
        assert df["load_mw"].max() <= 80000

    def test_load_forecast_timestamps(self):
        df = fetch_load_forecast(year=2023)
        assert df["timestamp"].iloc[0].year == 2023
        assert df["timestamp"].is_monotonic_increasing

    def test_generation_mix_shape(self):
        df = fetch_generation_mix(zone="FR", year=2023)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8760
        for col in ["nuclear", "gas", "wind", "solar", "hydro", "coal"]:
            assert col in df.columns

    def test_generation_mix_nonneg(self):
        df = fetch_generation_mix()
        for col in ["nuclear", "gas", "wind", "solar", "hydro", "coal"]:
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_generation_solar_zero_at_night(self):
        df = fetch_generation_mix()
        night = df[df["timestamp"].dt.hour == 3]
        assert (night["solar"] <= 1e-6).all()


# ── Preprocessing ─────────────────────────────────────────────────────

class TestPreprocess:
    @pytest.fixture
    def load_df(self):
        return fetch_load_forecast()

    def test_normalize_range(self, load_df):
        normed = normalize_load(load_df)
        assert "load_norm" in normed.columns
        assert normed["load_norm"].min() >= 0.0
        assert normed["load_norm"].max() <= 1.0

    def test_normalize_preserves_original(self, load_df):
        normed = normalize_load(load_df)
        assert "load_mw" in normed.columns
        np.testing.assert_array_equal(normed["load_mw"].values, load_df["load_mw"].values)

    def test_resample_hourly_identity(self, load_df):
        result = resample_hourly(load_df, freq="h")
        assert len(result) == len(load_df)

    def test_extract_horizon(self, load_df):
        h = extract_horizon(load_df, start_hour=100, n_hours=24)
        assert h.shape == (24,)
        assert h.dtype == np.float64

    def test_extract_horizon_clip(self, load_df):
        h = extract_horizon(load_df, start_hour=8750, n_hours=24)
        assert len(h) == 10  # 8760 - 8750

    def test_train_test_split(self, load_df):
        train, test = train_test_split_temporal(load_df, train_frac=0.8)
        assert len(train) + len(test) == len(load_df)
        assert len(train) == int(0.8 * len(load_df))

    def test_train_test_no_overlap(self, load_df):
        train, test = train_test_split_temporal(load_df)
        assert train["timestamp"].max() < test["timestamp"].min()


# ── Generator Fleet ───────────────────────────────────────────────────

class TestGenerator:
    def test_dataclass_fields(self):
        g = Generator(
            name="test", fuel_type="gas", capacity_mw=500,
            min_output_mw=100, fuel_cost=50, startup_cost=10000,
        )
        assert g.name == "test"
        assert g.capacity_mw == 500
        assert g.min_up_time == 1  # default

    def test_fleet_properties(self):
        fleet = build_fleet(n_generators=10)
        assert fleet.n_generators == 10
        assert fleet.total_capacity > 0

    def test_fleet_vectors(self):
        fleet = build_fleet(n_generators=6)
        assert fleet.cost_vector().shape == (6,)
        assert fleet.capacity_vector().shape == (6,)
        assert fleet.min_output_vector().shape == (6,)
        assert fleet.startup_cost_vector().shape == (6,)

    def test_fleet_to_dataframe(self):
        fleet = build_fleet(n_generators=8)
        df = fleet.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8
        assert "fuel_type" in df.columns

    def test_fleet_costs_positive(self):
        fleet = build_fleet(n_generators=10)
        assert (fleet.cost_vector() > 0).all()
        assert (fleet.startup_cost_vector() > 0).all()

    def test_fleet_min_less_than_max(self):
        fleet = build_fleet(n_generators=10)
        for g in fleet.generators:
            assert g.min_output_mw < g.capacity_mw, f"{g.name}: min >= max"

    def test_small_fleet(self):
        fleet = build_small_fleet(n=4)
        assert fleet.n_generators == 4
        for g in fleet.generators:
            assert g.min_up_time == 1
            assert g.min_down_time == 1
            assert g.ramp_rate == g.capacity_mw

    def test_small_fleet_sizes(self):
        for n in [2, 4, 6, 8]:
            fleet = build_small_fleet(n=n)
            assert fleet.n_generators == n

    def test_fleet_reproducible(self):
        f1 = build_fleet(n_generators=6, seed=99)
        f2 = build_fleet(n_generators=6, seed=99)
        np.testing.assert_array_equal(f1.cost_vector(), f2.cost_vector())

    def test_different_seeds_differ(self):
        f1 = build_fleet(n_generators=6, seed=1)
        f2 = build_fleet(n_generators=6, seed=2)
        assert not np.array_equal(f1.cost_vector(), f2.cost_vector())


# ── Integration ───────────────────────────────────────────────────────

class TestIntegration:
    def test_load_to_horizon(self):
        """Pipeline: download -> extract horizon -> check shape."""
        df = fetch_load_forecast()
        h = extract_horizon(df, start_hour=0, n_hours=6)
        assert h.shape == (6,)
        assert h.min() > 0

    def test_fleet_covers_demand(self):
        """Fleet total capacity should exceed typical demand."""
        df = fetch_load_forecast()
        peak = df["load_mw"].max()
        fleet = build_fleet(n_generators=10, total_capacity_mw=peak * 1.5)
        assert fleet.total_capacity > peak

    def test_project_imports(self):
        """All data submodules importable."""
        import data
        import data.download_entsoe
        import data.preprocess
        import data.generators
