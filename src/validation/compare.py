"""Inter-source validation: Open-Meteo vs NASA POWER vs IMD vs station."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.qc import metrics

COMPARE_COLS = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "rainfall_mm"]


def _align_hourly(left: pd.DataFrame, right: pd.DataFrame, left_col: str, right_col: str) -> tuple[np.ndarray, np.ndarray]:
    l = left[["timestamp", left_col]].copy()
    r = right[["timestamp", right_col]].copy()
    l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True).dt.floor("h")
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True).dt.floor("h")
    merged = l.merge(r, on="timestamp", suffixes=("_a", "_b"))
    a = pd.to_numeric(merged[f"{left_col}_a"], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(merged[f"{right_col}_b"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def compare_hourly_sources(
    source_a: pd.DataFrame,
    source_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    columns: tuple[str, ...] = tuple(COMPARE_COLS),
) -> list[dict]:
    scores: list[dict] = []
    for col in columns:
        if col not in source_a.columns or col not in source_b.columns:
            continue
        a, b = _align_hourly(source_a, source_b, col, col)
        if len(a) < 3:
            continue
        m = metrics(a, b)
        scores.append(
            {
                "source_a": name_a,
                "source_b": name_b,
                "channel": col,
                "granularity": "hourly",
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": m["bias"],
                "n": m["n"],
            }
        )
    return scores


def compare_daily_rain(
    hourly_a: pd.DataFrame,
    daily_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> dict | None:
    if hourly_a.empty or daily_b.empty:
        return None
    a = hourly_a.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"], utc=True)
    a["date"] = a["timestamp"].dt.date
    daily_a = a.groupby("date", as_index=False)["rainfall_mm"].sum(min_count=1)
    b = daily_b.copy()
    b["date"] = pd.to_datetime(b["date"]).dt.date
    merged = daily_a.merge(b[["date", "rainfall_mm"]], on="date", suffixes=("_a", "_b"))
    x = pd.to_numeric(merged["rainfall_mm_a"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(merged["rainfall_mm_b"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    m = metrics(x[ok], y[ok])
    return {
        "source_a": name_a,
        "source_b": name_b,
        "channel": "rainfall_mm",
        "granularity": "daily",
        **m,
    }
