"""Rolling-origin walk-forward backtests with lead-time breakdown."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.calibration.qc import metrics
from src.forecast.baselines import persistence_forecast, seasonal_naive_from_index
from src.forecast.features import prepare_hourly_series
from src.forecast.models import HourlyGradientBooster, TwoStageRainForecaster


def _skill_vs_naive(model_mae: float, naive_mae: float) -> float | None:
    if not np.isfinite(model_mae) or not np.isfinite(naive_mae) or naive_mae <= 0:
        return None
    return float(1.0 - model_mae / naive_mae)


def run_walkforward_backtest(
    frame: pd.DataFrame,
    value_col: str,
    channel: str,
    horizons: tuple[int, ...] = (6, 24, 72),
    step_hours: int = 24,
    min_train_hours: int = 48,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Rolling daily origins; for each origin train on prior history and score at fixed lead hours.
    Returns aggregated score rows (mean/std MAE by lead hour) and optional prediction samples.
    """
    hourly = prepare_hourly_series(frame, value_col).dropna(subset=["y"])
    if hourly.empty:
        raise ValueError(f"No hourly values for {value_col}")

    index = hourly.index.sort_values()
    start = index.min() + pd.Timedelta(hours=min_train_hours)
    end = index.max() - pd.Timedelta(hours=max(horizons))

    origins: list[pd.Timestamp] = []
    cursor = start.floor("h")
    while cursor <= end:
        origins.append(cursor)
        cursor += pd.Timedelta(hours=step_hours)

    if not origins:
        raise ValueError(f"Not enough history for walk-forward on {channel}")

    fitted_at = datetime.now(timezone.utc)
    model_errors: dict[int, dict[str, list[float]]] = {
        h: {"gradient_boosting": [], "persistence": [], "seasonal_naive": []} for h in horizons
    }
    pred_rows: list[dict] = []

    for origin in origins:
        train = hourly.loc[: origin - pd.Timedelta(hours=1)]
        if len(train) < min_train_hours:
            continue
        history = train["y"]
        booster: HourlyGradientBooster | TwoStageRainForecaster
        if channel == "rainfall_mm":
            booster = TwoStageRainForecaster()
        else:
            booster = HourlyGradientBooster()
        try:
            booster.fit(train)
        except ValueError:
            continue

        for lead in horizons:
            target_ts = origin + pd.Timedelta(hours=lead - 1)
            if target_ts not in hourly.index:
                continue
            actual = float(hourly.loc[target_ts, "y"])
            if not np.isfinite(actual):
                continue

            y_persist = float(persistence_forecast(history, lead)[-1])
            y_naive = float(
                seasonal_naive_from_index(history, pd.DatetimeIndex([target_ts]), season=24)[0]
            )
            y_gb = float(
                booster.predict_recursive(history, pd.DatetimeIndex([target_ts]))[0]
            )

            for model_name, pred in [
                ("persistence", y_persist),
                ("seasonal_naive", y_naive),
                ("gradient_boosting", y_gb),
            ]:
                err = abs(pred - actual)
                if np.isfinite(err):
                    model_errors[lead][model_name].append(err)
                    pred_rows.append(
                        {
                            "origin": origin,
                            "timestamp": target_ts,
                            "channel": channel,
                            "model": model_name,
                            "lead_hours": lead,
                            "predicted": pred,
                            "actual": actual,
                        }
                    )

    score_rows: list[dict] = []
    for lead in horizons:
        for model_name, errs in model_errors[lead].items():
            if not errs:
                continue
            arr = np.asarray(errs, dtype=float)
            mae = float(np.mean(arr))
            naive_mae = float(np.mean(model_errors[lead]["seasonal_naive"])) if model_errors[lead][
                "seasonal_naive"
            ] else float("nan")
            score_rows.append(
                {
                    "channel": channel,
                    "model": model_name,
                    "lead_hours": lead,
                    "mae": mae,
                    "mae_std": float(np.std(arr)),
                    "rmse": float(np.sqrt(np.mean(arr**2))),
                    "bias": float(np.mean(np.sign(arr) * arr)),  # approximate
                    "n": int(len(arr)),
                    "skill_vs_naive": _skill_vs_naive(mae, naive_mae)
                    if model_name != "seasonal_naive"
                    else None,
                    "fitted_at": fitted_at.isoformat(),
                    "train_mode": "walkforward",
                }
            )

    return pd.DataFrame(pred_rows), score_rows
