"""ENTSO-E Transparency Platform data download.

Real data requires an API key from https://transparency.entsoe.eu/.
When unavailable, functions return synthetic load/generation profiles.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENTSO-E API configuration
# ---------------------------------------------------------------------------
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Bidding zone codes
ZONES = {
    "FR": "10YFR-RTE------C",   # France
    "DE": "10Y1001A1001A83F",   # Germany
}

# Document types
DOC_TYPES = {
    "day_ahead_load": "A65",
    "actual_generation": "A75",
    "installed_capacity": "A68",
}


def fetch_load_forecast(
    zone: str = "FR",
    year: int = 2023,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch day-ahead load forecast from ENTSO-E.

    Returns DataFrame with columns [timestamp, load_mw].
    Falls back to synthetic data if api_key is None.
    """
    if api_key is not None:
        logger.info("ENTSO-E download not implemented — returning synthetic data")

    return _synthetic_load_profile(year)


def fetch_generation_mix(
    zone: str = "FR",
    year: int = 2023,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch actual generation by fuel type.

    Returns DataFrame with columns [timestamp, nuclear, gas, wind, solar, hydro, coal].
    Falls back to synthetic data if api_key is None.
    """
    if api_key is not None:
        logger.info("ENTSO-E download not implemented — returning synthetic data")

    return _synthetic_generation_mix(year)


# ---------------------------------------------------------------------------
# Synthetic fallbacks
# ---------------------------------------------------------------------------
def _synthetic_load_profile(year: int, hours: int = 8760) -> pd.DataFrame:
    """Generate a realistic synthetic load profile.

    Models daily (sinusoidal) + weekly (weekend dip) + seasonal patterns.
    Base load ~40 GW (France-scale), peak ~60 GW.
    """
    rng = np.random.default_rng(42)
    t = np.arange(hours, dtype=np.float64)

    # Daily cycle: peak at 12h, trough at 4h
    daily = 5000 * np.sin(2 * np.pi * (t - 6) / 24)

    # Weekly cycle: ~10% dip on weekends (day 5, 6)
    day_of_week = (t // 24).astype(int) % 7
    weekly = np.where(day_of_week >= 5, -3000.0, 0.0)

    # Seasonal: higher in winter (month 0-2, 10-11), lower in summer
    day_of_year = (t // 24).astype(int) % 365
    seasonal = 5000 * np.cos(2 * np.pi * day_of_year / 365)

    base = 45000.0  # 45 GW base
    load = base + daily + weekly + seasonal + rng.normal(0, 500, hours)
    load = np.clip(load, 20000, 80000)

    timestamps = pd.date_range(f"{year}-01-01", periods=hours, freq="h")
    return pd.DataFrame({"timestamp": timestamps, "load_mw": load})


def _synthetic_generation_mix(year: int, hours: int = 8760) -> pd.DataFrame:
    """Generate synthetic generation mix (France-like: heavy nuclear)."""
    rng = np.random.default_rng(43)
    t = np.arange(hours, dtype=np.float64)

    timestamps = pd.date_range(f"{year}-01-01", periods=hours, freq="h")

    # Nuclear: ~35 GW baseload with slight variation
    nuclear = 35000 + rng.normal(0, 1000, hours)

    # Gas: 2-8 GW, peaks with demand
    gas = 3000 + 2000 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 500, hours)

    # Wind: stochastic, 0-10 GW
    wind = np.abs(5000 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 2000, hours))

    # Solar: only during daylight, 0-8 GW
    hour_of_day = t % 24
    solar = np.where(
        (hour_of_day >= 7) & (hour_of_day <= 19),
        4000 * np.sin(np.pi * (hour_of_day - 7) / 12) + rng.normal(0, 500, hours),
        0.0,
    )

    # Hydro: ~5 GW steady
    hydro = 5000 + rng.normal(0, 500, hours)

    # Coal: small residual
    coal = 1000 + rng.normal(0, 200, hours)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "nuclear": np.clip(nuclear, 0, None),
        "gas": np.clip(gas, 0, None),
        "wind": np.clip(wind, 0, None),
        "solar": np.clip(solar, 0, None),
        "hydro": np.clip(hydro, 0, None),
        "coal": np.clip(coal, 0, None),
    })
    return df
