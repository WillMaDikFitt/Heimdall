"""Hold-out and walk-style forecast backtests against historical truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.calibration.qc import metrics
from src.forecast.baselines import persistence_forecast, seasonal_naive_from_index
from src.forecast.ensemble import bias_correct_series
from src.forecast.features import prepare_hourly_series
from src.forecast.metrics import (
    block_sum_mae,
    circular_mae_deg,
    daily_rain_metrics,
    lead_rain_metrics,
    rain_metrics,
)
from src.forecast.models import (
    DirectHorizonRainModel,
    HourlyGradientBooster,
    TwoStageRainForecaster,
    allocate_to_sum,
)

RAIN_EXO_COLS = (
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
    "rainfall_mm",
)


@dataclass
class BacktestSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp


def make_holdout_split(
    index: pd.DatetimeIndex,
    train_days: int,
    forecast_days: int,
) -> BacktestSplit:
    """
    Use the last (train_days + forecast_days) of the series:
    first train_days -> fit, next forecast_days -> score against held-out truth.
    """
    if index.empty:
        raise ValueError("Empty series index")
    end = index.max().floor("h")
    forecast_start = end - pd.Timedelta(days=forecast_days) + pd.Timedelta(hours=1)
    train_end = forecast_start - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=train_days) + pd.Timedelta(hours=1)
    if train_start < index.min():
        train_start = index.min().ceil("h")
    return BacktestSplit(
        train_start=train_start,
        train_end=train_end,
        forecast_start=forecast_start,
        forecast_end=end,
    )


def make_extended_train_split(
    index: pd.DatetimeIndex,
    forecast_days: int,
) -> BacktestSplit:
    """Train on all history before the final forecast_days window."""
    end = index.max().floor("h")
    forecast_start = end - pd.Timedelta(days=forecast_days) + pd.Timedelta(hours=1)
    train_end = forecast_start - pd.Timedelta(hours=1)
    train_start = index.min().ceil("h")
    return BacktestSplit(train_start, train_end, forecast_start, end)


def _prepare_exogenous(
    exogenous: pd.DataFrame | None,
    value_col: str,
    index: pd.DatetimeIndex,
    channel: str | None = None,
) -> pd.DataFrame | None:
    if exogenous is None or exogenous.empty:
        return None
    exo = exogenous.copy()
    exo["timestamp"] = pd.to_datetime(exo["timestamp"], utc=True)
    exo = exo.set_index("timestamp").sort_index()
    cols: list[str] = []
    if value_col in exo.columns:
        cols.append(value_col)
    if channel == "rainfall_mm":
        for name in RAIN_EXO_COLS:
            if name in exo.columns and name not in cols:
                cols.append(name)
        for name in exo.columns:
            if str(name).startswith("nwp_") and name not in cols:
                cols.append(name)
    if not cols:
        return None
    out = exo[cols].apply(pd.to_numeric, errors="coerce").reindex(index)
    out = out.mask(out <= -900.0)
    return out


def _attach_rain_scores(
    row: dict,
    forecast_index: pd.DatetimeIndex,
    yhat: np.ndarray,
    actual: np.ndarray,
) -> None:
    rain = rain_metrics(yhat, actual)
    row["csi"] = rain["csi"]
    row["wet_hour_bias"] = rain["wet_hour_bias"]
    row["hit_rate"] = rain["hit_rate"]
    row["false_alarm_rate"] = rain.get("false_alarm_rate")
    row["wet_hours_pred"] = rain.get("wet_hours_pred")
    row["wet_hours_act"] = rain.get("wet_hours_act")
    daily = daily_rain_metrics(forecast_index, yhat, actual)
    row["daily_mae"] = daily["daily_mae"]
    row["daily_bias"] = daily["daily_bias"]
    row["block_24h_mae"] = block_sum_mae(yhat, actual, block_hours=24)
    row["block_6h_mae"] = block_sum_mae(yhat, actual, block_hours=6)
    row.update(lead_rain_metrics(yhat, actual, lead_hours=6))
    row.update(lead_rain_metrics(yhat, actual, lead_hours=24))


def run_channel_backtest(
    frame: pd.DataFrame,
    value_col: str,
    channel: str,
    train_days: int = 3,
    forecast_days: int = 3,
    use_full_history_train: bool = False,
    exogenous: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Return (predictions long-frame, score rows).

    Models compared:
    - persistence
    - seasonal_naive (24h)
    - gradient_boosting (recursive; two-stage for rain)
    - bias_corrected_ref (when a separate reference series is available)
    - hybrid_rain / direct_rain_24h (rain only)
    """
    hourly = prepare_hourly_series(frame, value_col)
    hourly = hourly.dropna(subset=["y"])
    if hourly.empty:
        raise ValueError(f"No hourly values for {value_col}")

    split = (
        make_extended_train_split(hourly.index, forecast_days)
        if use_full_history_train
        else make_holdout_split(hourly.index, train_days, forecast_days)
    )

    train = hourly.loc[split.train_start : split.train_end]
    actual = hourly.loc[split.forecast_start : split.forecast_end]
    if train.empty or actual.empty:
        raise ValueError(
            f"Bad split for {channel}: train={len(train)} actual={len(actual)}. "
            "Need more historical hours."
        )

    forecast_index = actual.index
    history = train["y"]
    horizon = len(forecast_index)
    # Exogenous needs train + forecast hours so horizon rain sums are available.
    exo_full = _prepare_exogenous(
        exogenous,
        value_col,
        train.index.union(forecast_index),
        channel=channel,
    )
    exo_train = exo_full.reindex(train.index) if exo_full is not None else None
    exo_fc = exo_full.reindex(forecast_index) if exo_full is not None else None

    preds: dict[str, np.ndarray] = {
        "persistence": persistence_forecast(history, horizon),
        "seasonal_naive": seasonal_naive_from_index(history, forecast_index, season=24),
    }

    if channel == "rainfall_mm":
        two_stage = TwoStageRainForecaster()
        two_stage.fit(train, exogenous=exo_train)
        two_stage_hourly = two_stage.predict_recursive(
            history, forecast_index, exogenous=exo_fc
        )
        preds["gradient_boosting"] = two_stage_hourly

        # Timing shape: prefer two-stage; fall back to NWP mean / reference rain.
        shape = two_stage_hourly.copy()
        if exo_fc is not None:
            if "nwp_rain_mean" in exo_fc.columns:
                exo_shape = pd.to_numeric(exo_fc["nwp_rain_mean"], errors="coerce").fillna(0.0).to_numpy()
            elif "rainfall_mm" in exo_fc.columns:
                exo_shape = pd.to_numeric(exo_fc["rainfall_mm"], errors="coerce").fillna(0.0).to_numpy()
            else:
                exo_shape = None
            if exo_shape is not None and float(np.nansum(shape)) <= 1e-9 and float(np.nansum(exo_shape)) > 0:
                shape = exo_shape

        try:
            direct24 = DirectHorizonRainModel(horizon_hours=24)
            direct24.fit(train, exogenous=exo_full)
            hybrid = direct24.predict_block_profile(
                history,
                forecast_index,
                shape=shape,
                exogenous=exo_full,
                block_hours=24,
            )
            preds["hybrid_rain"] = hybrid

            if exo_fc is not None and "nwp_rain_mean" in exo_fc.columns:
                nwp_shape = pd.to_numeric(exo_fc["nwp_rain_mean"], errors="coerce").fillna(0.0).to_numpy()
            elif exo_fc is not None and "rainfall_mm" in exo_fc.columns:
                nwp_shape = pd.to_numeric(exo_fc["rainfall_mm"], errors="coerce").fillna(0.0).to_numpy()
            else:
                nwp_shape = np.ones(horizon, dtype=float)
            preds["direct_rain_24h"] = direct24.predict_block_profile(
                history,
                forecast_index,
                shape=nwp_shape,
                exogenous=exo_full,
                block_hours=24,
            )
        except ValueError:
            # Short windows can fail direct training; keep two-stage only.
            pass

        try:
            direct6 = DirectHorizonRainModel(horizon_hours=6)
            direct6.fit(train, exogenous=exo_full)
            first6 = min(6, horizon)
            total6 = direct6.predict_sum(
                history,
                pd.Timestamp(forecast_index[0]),
                exogenous=exo_full,
            )
            profile6 = np.zeros(horizon, dtype=float)
            profile6[:first6] = allocate_to_sum(shape[:first6], total6)
            # Beyond 6h keep hybrid/two-stage timing if already computed.
            if "hybrid_rain" in preds:
                profile6[first6:] = preds["hybrid_rain"][first6:]
            else:
                profile6[first6:] = two_stage_hourly[first6:]
            preds["direct_rain_6h"] = profile6
        except ValueError:
            pass
    else:
        booster = HourlyGradientBooster()
        booster.fit(train, exogenous=exo_train)
        preds["gradient_boosting"] = booster.predict_recursive(
            history, forecast_index, exogenous=exo_fc
        )

    # Bias-correct a reference / Open-Meteo series when it is available as exogenous.
    if (
        exogenous is not None
        and not exogenous.empty
        and value_col in exogenous.columns
        and exo_full is not None
        and value_col in exo_full.columns
    ):
        corrected = bias_correct_series(
            exo_full[value_col],
            hourly["y"],
            train.index,
        )
        preds["bias_corrected_ref"] = corrected.reindex(forecast_index).to_numpy(dtype=float)

    fitted_at = datetime.now(timezone.utc)
    long_rows: list[dict] = []
    score_rows: list[dict] = []
    actual_y = actual["y"].to_numpy(dtype=float)

    for model_name, yhat in preds.items():
        score = metrics(yhat, actual_y)
        row = {
            "channel": channel,
            "model": model_name,
            "train_start": split.train_start.isoformat(),
            "train_end": split.train_end.isoformat(),
            "forecast_start": split.forecast_start.isoformat(),
            "forecast_end": split.forecast_end.isoformat(),
            "train_hours": int(len(train)),
            "forecast_hours": int(horizon),
            "mae": score["mae"],
            "rmse": score["rmse"],
            "bias": score["bias"],
            "n": score["n"],
            "fitted_at": fitted_at.isoformat(),
            "train_mode": "full_history" if use_full_history_train else f"last_{train_days}d",
            "csi": None,
            "wet_hour_bias": None,
            "hit_rate": None,
            "daily_mae": None,
            "block_24h_mae": None,
            "block_6h_mae": None,
            "csi_6h": None,
            "csi_24h": None,
            "hit_rate_6h": None,
            "hit_rate_24h": None,
        }
        if channel == "rainfall_mm":
            _attach_rain_scores(row, forecast_index, yhat, actual_y)
        if channel == "wind_direction_deg":
            row["circular_mae"] = circular_mae_deg(yhat, actual_y)
        score_rows.append(row)
        for ts, pred, truth in zip(forecast_index, yhat, actual_y, strict=True):
            long_rows.append(
                {
                    "timestamp": ts,
                    "channel": channel,
                    "model": model_name,
                    "predicted": float(pred) if np.isfinite(pred) else None,
                    "actual": float(truth) if np.isfinite(truth) else None,
                }
            )

    return pd.DataFrame(long_rows), score_rows
