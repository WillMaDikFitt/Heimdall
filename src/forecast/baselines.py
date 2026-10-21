"""Simple forecasting baselines for backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every step in the horizon."""
    last = float(history.dropna().iloc[-1])
    return np.full(horizon, last, dtype=float)


def seasonal_naive_from_index(
    history: pd.Series,
    forecast_index: pd.DatetimeIndex,
    season: int = 24,
) -> np.ndarray:
    """Map each forecast timestamp to history[t - season], falling back to last value."""
    hist = history.dropna()
    last = float(hist.iloc[-1]) if not hist.empty else float("nan")
    preds: list[float] = []
    for ts in forecast_index:
        prev = ts - pd.Timedelta(hours=season)
        if prev in hist.index and np.isfinite(hist.loc[prev]):
            preds.append(float(hist.loc[prev]))
        else:
            candidates = hist.loc[:prev]
            preds.append(float(candidates.iloc[-1]) if not candidates.empty else last)
    return np.asarray(preds, dtype=float)
