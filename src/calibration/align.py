"""Timestamp alignment and resampling of sensor vs hourly reference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import CHANNEL_NAMES


def circular_mean(degrees: pd.Series) -> float:
    values = pd.to_numeric(degrees, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    rad = np.deg2rad(values.to_numpy())
    mean_sin = np.sin(rad).mean()
    mean_cos = np.cos(rad).mean()
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)


def resample_sensor(readings: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    if readings.empty:
        return readings
    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    freq = f"{minutes}min"

    grouped = df[list(CHANNEL_NAMES)].resample(freq)
    out = grouped.mean()
    if "rainfall_mm" in df.columns:
        out["rainfall_mm"] = grouped["rainfall_mm"].sum()
    if "wind_direction_deg" in df.columns:
        out["wind_direction_deg"] = grouped["wind_direction_deg"].apply(circular_mean)
    extra = [c for c in ("station_id", "source") if c in df.columns]
    if extra:
        meta = df[extra].resample(freq).first()
        out = out.join(meta)
    return out.reset_index()


def align_to_reference(
    sensor_hourly: pd.DataFrame,
    reference: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge resampled sensor with reference. Sensor columns get `_raw` suffix."""
    left = sensor_hourly.copy()
    right = reference.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)

    left = left.set_index("timestamp")
    right = right.set_index("timestamp")
    left.index = left.index.floor("h")
    right.index = right.index.floor("h")

    raw = left[list(CHANNEL_NAMES)].add_suffix("_raw")
    ref = right[list(CHANNEL_NAMES)].add_suffix("_ref")
    merged = raw.join(ref, how=how)
    if "station_id" in left.columns:
        merged = merged.join(left[["station_id"]])
    return merged.reset_index()
