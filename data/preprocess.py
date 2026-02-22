"""Preprocessing for grid data — normalization, resampling, train/test split."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def normalize_load(df: pd.DataFrame, col: str = "load_mw") -> pd.DataFrame:
    """Min-max normalize load column to [0, 1]. Adds 'load_norm' column."""
    out = df.copy()
    lo, hi = out[col].min(), out[col].max()
    if hi - lo < 1e-9:
        out["load_norm"] = 0.5
    else:
        out["load_norm"] = (out[col] - lo) / (hi - lo)
    return out


def resample_hourly(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Resample timestamped DataFrame to given frequency (default hourly)."""
    out = df.copy()
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    out = out.resample(freq).mean().interpolate()
    out = out.reset_index()
    return out


def extract_horizon(
    df: pd.DataFrame,
    start_hour: int = 0,
    n_hours: int = 24,
) -> np.ndarray:
    """Extract demand vector for a planning horizon.

    Returns shape (n_hours,) array of load values in MW.
    """
    col = "load_mw" if "load_mw" in df.columns else df.columns[1]
    values = df[col].values
    end = min(start_hour + n_hours, len(values))
    return values[start_hour:end].astype(np.float64)


def train_test_split_temporal(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame temporally (first train_frac for train, rest for test)."""
    n = len(df)
    split = int(n * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()
