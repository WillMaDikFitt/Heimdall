"""Lag and calendar features for hourly weather forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48)
DEFAULT_ROLLING = (3, 6, 24)


def prepare_hourly_series(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return a clean hourly series with UTC timestamp index."""
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    series = pd.to_numeric(df[value_col], errors="coerce")
    # NASA POWER and some archives use large negative fill codes for missing.
    series = series.mask(series <= -900.0)
    out = series.to_frame(name="y")
    out = out.resample("1h").mean()
    return out


def add_features(
    series: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling: tuple[int, ...] = DEFAULT_ROLLING,
    exogenous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build supervised learning features from an hourly target column `y`."""
    df = series.copy()
    if "y" not in df.columns:
        raise ValueError("Expected column 'y'")

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in rolling:
        df[f"roll_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["y"].shift(1).rolling(window).std()

    if exogenous is not None and not exogenous.empty:
        exo = exogenous.copy()
        exo.index = pd.to_datetime(exo.index, utc=True)
        for col in exo.columns:
            df[f"exo_{col}"] = exo[col].reindex(df.index)

    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "y"]
