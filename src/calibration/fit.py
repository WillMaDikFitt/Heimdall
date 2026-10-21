"""Linear per-channel calibration: corrected = scale * raw + offset."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.calibration.qc import iqr_inlier_mask, metrics, quality_flags
from src.schema import CHANNEL_NAMES


def _clip_calibrated(channel: str, values: np.ndarray) -> np.ndarray:
    if channel == "humidity_pct":
        return np.clip(values, 0, 100)
    if channel in {"rainfall_mm", "wind_speed_ms"}:
        return np.clip(values, 0, None)
    if channel == "wind_direction_deg":
        return np.mod(values, 360.0)
    return values


def split_fit_holdout(
    aligned: pd.DataFrame,
    holdout_fraction: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split: earlier rows for fit, trailing rows for hold-out."""
    if aligned.empty:
        return aligned, aligned
    df = aligned.sort_values("timestamp").reset_index(drop=True)
    fraction = float(np.clip(holdout_fraction, 0.05, 0.5))
    split_idx = max(8, int(len(df) * (1.0 - fraction)))
    if split_idx >= len(df):
        split_idx = max(1, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def fit_channel(
    raw: np.ndarray,
    reference: np.ndarray,
    iqr_multiplier: float = 1.5,
) -> dict:
    pair_ok = np.isfinite(raw) & np.isfinite(reference)
    residuals = np.where(pair_ok, raw - reference, np.nan)
    inliers = iqr_inlier_mask(residuals, multiplier=iqr_multiplier) & pair_ok
    n = int(inliers.sum())

    if n < 8 or float(np.nanstd(raw[inliers])) < 1e-6:
        scale, offset = 1.0, 0.0
    else:
        model = LinearRegression()
        model.fit(raw[inliers].reshape(-1, 1), reference[inliers])
        scale = float(model.coef_[0])
        offset = float(model.intercept_)

    return {
        "scale": scale,
        "offset": offset,
        "n_points": n,
        "inliers": inliers,
        "calibrated": scale * raw + offset,
        "pair_ok": pair_ok,
    }


def _score_subset(raw: np.ndarray, ref: np.ndarray, scale: float, offset: float, channel: str) -> dict:
    cal = _clip_calibrated(channel, scale * raw + offset)
    return {
        "mae_cal": metrics(cal, ref)["mae"],
        "rmse_cal": metrics(cal, ref)["rmse"],
        "bias_cal": metrics(cal, ref)["bias"],
        "mae_raw": metrics(raw, ref)["mae"],
        "rmse_raw": metrics(raw, ref)["rmse"],
        "bias_raw": metrics(raw, ref)["bias"],
    }


def fit_all_channels(
    aligned: pd.DataFrame,
    station_id: str,
    iqr_multiplier: float = 1.5,
    holdout_fraction: float = 0.3,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fit coefficients on the earlier time window; report in-sample and hold-out MAE separately.
    """
    fitted_at = datetime.now(timezone.utc)
    fit_frame, holdout_frame = split_fit_holdout(aligned, holdout_fraction=holdout_fraction)
    long_rows: list[dict] = []
    coeff_rows: list[dict] = []

    for channel in CHANNEL_NAMES:
        raw_fit = pd.to_numeric(fit_frame[f"{channel}_raw"], errors="coerce").to_numpy(dtype=float)
        ref_fit = pd.to_numeric(fit_frame[f"{channel}_ref"], errors="coerce").to_numpy(dtype=float)
        result = fit_channel(raw_fit, ref_fit, iqr_multiplier=iqr_multiplier)

        raw_all = pd.to_numeric(aligned[f"{channel}_raw"], errors="coerce").to_numpy(dtype=float)
        ref_all = pd.to_numeric(aligned[f"{channel}_ref"], errors="coerce").to_numpy(dtype=float)
        calibrated_all = _clip_calibrated(channel, result["scale"] * raw_all + result["offset"])

        # Recompute inlier mask on full series for quality flags
        pair_ok = np.isfinite(raw_all) & np.isfinite(ref_all)
        residuals = np.where(pair_ok, raw_all - ref_all, np.nan)
        inliers = iqr_inlier_mask(residuals, multiplier=iqr_multiplier) & pair_ok
        flags = quality_flags(raw_all, ref_all, calibrated_all, inliers)

        in_sample = _score_subset(raw_fit, ref_fit, result["scale"], result["offset"], channel)
        holdout: dict[str, float | None] = {
            "mae_holdout": None,
            "rmse_holdout": None,
            "bias_holdout": None,
            "n_holdout": 0,
        }
        if not holdout_frame.empty:
            raw_ho = pd.to_numeric(holdout_frame[f"{channel}_raw"], errors="coerce").to_numpy(
                dtype=float
            )
            ref_ho = pd.to_numeric(holdout_frame[f"{channel}_ref"], errors="coerce").to_numpy(
                dtype=float
            )
            ho = _score_subset(raw_ho, ref_ho, result["scale"], result["offset"], channel)
            holdout = {
                "mae_holdout": ho["mae_cal"],
                "rmse_holdout": ho["rmse_cal"],
                "bias_holdout": ho["bias_cal"],
                "n_holdout": int(np.isfinite(raw_ho).sum()),
            }

        coeff_rows.append(
            {
                "station_id": station_id,
                "channel": channel,
                "scale": result["scale"],
                "offset": result["offset"],
                "fitted_at": fitted_at,
                "n_points": result["n_points"],
                "mae": in_sample["mae_cal"],
                "rmse": in_sample["rmse_cal"],
                "bias": in_sample["bias_cal"],
                "mae_raw": in_sample["mae_raw"],
                "rmse_raw": in_sample["rmse_raw"],
                "bias_raw": in_sample["bias_raw"],
                **holdout,
            }
        )

        for ts, raw_v, cal_v, ref_v, flag in zip(
            aligned["timestamp"], raw_all, calibrated_all, ref_all, flags, strict=True
        ):
            ts_out = pd.Timestamp(ts)
            if ts_out.tzinfo is None:
                ts_out = ts_out.tz_localize("UTC")
            long_rows.append(
                {
                    "timestamp": ts_out,
                    "station_id": station_id,
                    "channel": channel,
                    "raw": None if not np.isfinite(raw_v) else float(raw_v),
                    "calibrated": None if not np.isfinite(cal_v) else float(cal_v),
                    "reference_value": None if not np.isfinite(ref_v) else float(ref_v),
                    "quality_flag": flag,
                }
            )

    return pd.DataFrame(long_rows), coeff_rows
