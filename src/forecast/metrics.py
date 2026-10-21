"""Rain-aware and wind-direction forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rain_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    wet_threshold_mm: float = 0.1,
) -> dict:
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(act)
    pred, act = pred[ok], act[ok]
    if len(pred) == 0:
        return {
            "wet_hour_bias": float("nan"),
            "csi": float("nan"),
            "hit_rate": float("nan"),
            "false_alarm_rate": float("nan"),
            "wet_hours_pred": 0,
            "wet_hours_act": 0,
            "rain_n": 0,
        }

    wet_pred = pred >= wet_threshold_mm
    wet_act = act >= wet_threshold_mm
    hits = int(np.sum(wet_pred & wet_act))
    misses = int(np.sum(~wet_pred & wet_act))
    false_alarms = int(np.sum(wet_pred & ~wet_act))
    denom = hits + misses + false_alarms
    csi = float(hits / denom) if denom > 0 else float("nan")
    hit_rate = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    false_alarm_rate = (
        float(false_alarms / (false_alarms + hits)) if (false_alarms + hits) > 0 else float("nan")
    )

    wet_mask = wet_act
    if wet_mask.any():
        wet_hour_bias = float(np.mean(pred[wet_mask] - act[wet_mask]))
    else:
        wet_hour_bias = float("nan")

    return {
        "wet_hour_bias": wet_hour_bias,
        "csi": csi,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "wet_hours_pred": int(wet_pred.sum()),
        "wet_hours_act": int(wet_act.sum()),
        "rain_n": int(len(pred)),
    }


def lead_rain_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    lead_hours: int,
    wet_threshold_mm: float = 0.1,
) -> dict:
    """CSI / hit-rate on the first `lead_hours` of a forecast window."""
    n = max(0, min(int(lead_hours), len(predicted), len(actual)))
    metrics = rain_metrics(predicted[:n], actual[:n], wet_threshold_mm=wet_threshold_mm)
    return {
        f"csi_{lead_hours}h": metrics["csi"],
        f"hit_rate_{lead_hours}h": metrics["hit_rate"],
        f"wet_hours_pred_{lead_hours}h": metrics["wet_hours_pred"],
        f"wet_hours_act_{lead_hours}h": metrics["wet_hours_act"],
    }


def block_sum_mae(
    predicted: np.ndarray,
    actual: np.ndarray,
    block_hours: int = 24,
) -> float:
    """Mean absolute error on non-overlapping block totals (e.g. daily rain)."""
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    n = min(len(pred), len(act))
    if n < block_hours:
        if n == 0:
            return float("nan")
        return float(abs(np.nansum(pred[:n]) - np.nansum(act[:n])))
    blocks = n // block_hours
    if blocks <= 0:
        return float("nan")
    errs: list[float] = []
    for i in range(blocks):
        start = i * block_hours
        end = start + block_hours
        errs.append(abs(float(np.nansum(pred[start:end]) - np.nansum(act[start:end]))))
    return float(np.mean(errs)) if errs else float("nan")


def daily_rain_metrics(
    timestamps: pd.DatetimeIndex | np.ndarray,
    predicted: np.ndarray,
    actual: np.ndarray,
) -> dict:
    """Compare calendar-day rain totals (UTC date)."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "predicted": np.asarray(predicted, dtype=float),
            "actual": np.asarray(actual, dtype=float),
        }
    ).dropna(subset=["predicted", "actual"])
    if frame.empty:
        return {"daily_mae": float("nan"), "daily_bias": float("nan"), "n_days": 0}
    frame["date"] = frame["timestamp"].dt.floor("D")
    daily = frame.groupby("date", as_index=False).agg(
        predicted=("predicted", "sum"),
        actual=("actual", "sum"),
    )
    delta = daily["predicted"] - daily["actual"]
    return {
        "daily_mae": float(delta.abs().mean()),
        "daily_bias": float(delta.mean()),
        "n_days": int(len(daily)),
    }


def circular_mae_deg(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(act)
    if ok.sum() == 0:
        return float("nan")
    diff = np.abs(((pred[ok] - act[ok] + 180.0) % 360.0) - 180.0)
    return float(np.mean(diff))
