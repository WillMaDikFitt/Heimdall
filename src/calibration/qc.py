"""Outlier masks, quality flags, and error metrics vs reference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_inlier_mask(residuals: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    values = residuals.astype(float)
    finite = np.isfinite(values)
    mask = np.zeros(values.shape, dtype=bool)
    if finite.sum() < 4:
        mask[finite] = True
        return mask
    q1, q3 = np.nanpercentile(values[finite], [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        mask[finite] = True
        return mask
    low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
    mask[finite] = (values[finite] >= low) & (values[finite] <= high)
    return mask


def quality_flags(
    raw: np.ndarray,
    reference: np.ndarray,
    calibrated: np.ndarray,
    inliers: np.ndarray,
) -> list[str]:
    flags: list[str] = []
    for i in range(len(raw)):
        if not np.isfinite(raw[i]):
            flags.append("missing")
        elif not inliers[i] or not np.isfinite(reference[i]):
            flags.append("suspect")
        elif not np.isfinite(calibrated[i]):
            flags.append("suspect")
        else:
            flags.append("good")
    return flags


def metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    if ok.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": 0}
    err = pred[ok] - ref[ok]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "n": int(ok.sum()),
    }


def window_metrics(
    frame: pd.DataFrame,
    hours: int,
    value_col: str = "calibrated",
    ref_col: str = "reference_value",
) -> dict[str, float]:
    if frame.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": 0}
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    cutoff = ts.max() - pd.Timedelta(hours=hours)
    subset = frame.loc[ts >= cutoff]
    return metrics(subset[value_col].to_numpy(), subset[ref_col].to_numpy())
