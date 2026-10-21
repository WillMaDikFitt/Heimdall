"""Multi-model ensemble mean/spread and short-window bias correction."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import load_config
from src.forecast.providers.ecmwf_open import fetch_ecmwf
from src.forecast.providers.gfs import fetch_gfs
from src.forecast.providers.met_norway import fetch_met_norway

MEMBER_FETCHERS = {
    "met_norway": fetch_met_norway,
    "gfs": fetch_gfs,
    "ecmwf": fetch_ecmwf,
}

SCALAR_COLS = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms"]
RAIN_COL = "rainfall_mm"
WET_THRESHOLD_MM = 0.1
RAIN_SCALE_MIN = 0.4
RAIN_SCALE_MAX = 1.6


def fetch_all_members(cfg: dict | None = None, forecast_days: int = 7) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    members = list(cfg.get("forecast", {}).get("ensemble_members", MEMBER_FETCHERS.keys()))
    out: dict[str, pd.DataFrame] = {}
    for name in members:
        fetcher = MEMBER_FETCHERS.get(name)
        if fetcher is None:
            continue
        try:
            if name == "met_norway":
                out[name] = fetcher(cfg=cfg)
            else:
                out[name] = fetcher(cfg=cfg, forecast_days=forecast_days)
        except Exception:
            continue
    return out


def _aggregate_column(frames: list[pd.Series], col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    mat = pd.concat(frames, axis=1)
    if col == RAIN_COL:
        mean = mat.median(axis=1, skipna=True)
    else:
        mean = mat.mean(axis=1, skipna=True)
    spread = mat.std(axis=1, skipna=True)
    return mean, spread, mat.count(axis=1)


def build_ensemble(members: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean for scalars; median for rain; spread as uncertainty."""
    if not members:
        return pd.DataFrame()

    aligned: dict[str, pd.DataFrame] = {}
    for name, df in members.items():
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        aligned[name] = d.set_index("timestamp").sort_index()

    index = sorted(set().union(*(df.index for df in aligned.values())))
    index = pd.DatetimeIndex(index)
    rows: list[dict] = []
    for ts in index:
        row: dict = {"timestamp": ts}
        for col in SCALAR_COLS + [RAIN_COL]:
            series_list = []
            for df in aligned.values():
                if col in df.columns and ts in df.index:
                    val = df.at[ts, col]
                    if pd.notna(val):
                        series_list.append(pd.Series({ts: float(val)}))
            if not series_list:
                row[col] = np.nan
                row[f"{col}_spread"] = np.nan
                row[f"{col}_n_members"] = 0
                continue
            mean, spread, n = _aggregate_column(series_list, col)
            row[col] = float(mean.iloc[0]) if len(mean) else np.nan
            row[f"{col}_spread"] = float(spread.iloc[0]) if len(spread) else np.nan
            row[f"{col}_n_members"] = int(n.iloc[0]) if len(n) else 0
        rows.append(row)

    out = pd.DataFrame(rows)
    out["fetched_at"] = datetime.now(timezone.utc)
    out["source"] = "ensemble:" + "+".join(sorted(members.keys()))
    return out


def ensemble_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bias-corrected ensemble hourly rows to daily outlook cards."""
    def daily_rain_chance(series: pd.Series) -> float:
        rain = pd.to_numeric(series, errors="coerce").fillna(0.0)
        if rain.empty:
            return 0.0
        # Share of hours with measurable rain, expressed as a chance-style percent.
        return float(np.clip(100.0 * (rain >= 0.1).mean(), 0, 100))

    df = hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date", as_index=False).agg(
        temperature_max_c=("temperature_c", "max"),
        temperature_min_c=("temperature_c", "min"),
        rainfall_sum_mm=("rainfall_mm", "sum"),
        wind_speed_max_ms=("wind_speed_ms", "max"),
        rain_probability_max_pct=("rainfall_mm", daily_rain_chance),
    )
    daily["weather_code"] = 3  # neutral overcast placeholder
    daily["sunrise"] = None
    daily["sunset"] = None
    return daily


def ensemble_as_operational(
    hourly: pd.DataFrame,
    probability_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Prepare ensemble hourly rows for dashboard charts.

    Prefer real Open-Meteo rain probabilities when available; otherwise estimate
    chance from whether the ensemble expects measurable rain that hour.
    """
    out = hourly.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    rain = pd.to_numeric(out.get("rainfall_mm"), errors="coerce").fillna(0.0)

    if probability_source is not None and not probability_source.empty:
        src = probability_source.copy()
        src["timestamp"] = pd.to_datetime(src["timestamp"], utc=True)
        if "rain_probability_pct" in src.columns:
            merged = out.merge(
                src[["timestamp", "rain_probability_pct"]].rename(
                    columns={"rain_probability_pct": "rain_probability_pct_src"}
                ),
                on="timestamp",
                how="left",
            )
            out["rain_probability_pct"] = pd.to_numeric(
                merged["rain_probability_pct_src"], errors="coerce"
            )

    if "rain_probability_pct" not in out.columns:
        out["rain_probability_pct"] = np.nan

    missing = out["rain_probability_pct"].isna()
    # Honest fallback: if models expect rain this hour, treat chance as high; else low.
    out.loc[missing, "rain_probability_pct"] = np.where(rain[missing] >= 0.1, 70.0, 15.0)
    out["rain_probability_pct"] = np.clip(
        pd.to_numeric(out["rain_probability_pct"], errors="coerce").fillna(0.0), 0, 100
    )
    if "weather_code" not in out.columns:
        out["weather_code"] = np.where(rain >= 2.0, 63, np.where(rain >= 0.1, 61, 3))
    return out


def estimate_channel_biases(
    truth: pd.DataFrame,
    hindcast: pd.DataFrame,
    window_days: int = 14,
    end_at: pd.Timestamp | None = None,
    channels: list[str] | None = None,
) -> dict[str, float]:
    """
    Estimate mean bias (truth - hindcast) over a recent overlapping window.

    `hindcast` is a past model series (for example Open-Meteo history) that
    overlaps with `truth`. The resulting biases can be added to a live ensemble.
    """
    if truth.empty or hindcast.empty:
        return {}

    channels = channels or (SCALAR_COLS + [RAIN_COL])
    truth = truth.copy()
    hindcast = hindcast.copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"], utc=True)
    hindcast["timestamp"] = pd.to_datetime(hindcast["timestamp"], utc=True)
    # Drop common missing fill codes before bias math.
    for frame in (truth, hindcast):
        for col in channels:
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce")
                frame[col] = values.mask(values <= -900.0)

    end = end_at or min(truth["timestamp"].max(), hindcast["timestamp"].max())
    start = end - pd.Timedelta(days=window_days)
    t_win = truth.loc[(truth["timestamp"] >= start) & (truth["timestamp"] <= end)]
    h_win = hindcast.loc[(hindcast["timestamp"] >= start) & (hindcast["timestamp"] <= end)]
    if t_win.empty or h_win.empty:
        return {}

    biases: dict[str, float] = {}
    for col in channels:
        if col not in t_win.columns or col not in h_win.columns:
            continue
        merged = t_win[["timestamp", col]].merge(
            h_win[["timestamp", col]],
            on="timestamp",
            suffixes=("_truth", "_fc"),
        )
        if merged.empty:
            continue
        delta = pd.to_numeric(merged[f"{col}_truth"], errors="coerce") - pd.to_numeric(
            merged[f"{col}_fc"], errors="coerce"
        )
        delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
        if len(delta) < 12:
            continue
        biases[col] = float(delta.mean())
    return biases


def estimate_rain_scale(
    truth: pd.DataFrame,
    hindcast: pd.DataFrame,
    window_days: int = 14,
    end_at: pd.Timestamp | None = None,
) -> float | None:
    """
    Daily-total rain scale: mean(truth day sum) / mean(hindcast day sum).

    Used instead of an hourly millimetre add, which turns dry hours wet and
    inflates reservoir totals. Returns None when the window is too thin.
    """
    if truth.empty or hindcast.empty:
        return None
    if RAIN_COL not in truth.columns or RAIN_COL not in hindcast.columns:
        return None

    t = truth.copy()
    h = hindcast.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    for frame in (t, h):
        values = pd.to_numeric(frame[RAIN_COL], errors="coerce")
        frame[RAIN_COL] = values.mask(values <= -900.0).clip(lower=0.0)

    end = end_at or min(t["timestamp"].max(), h["timestamp"].max())
    start = end - pd.Timedelta(days=window_days)
    t = t.loc[(t["timestamp"] >= start) & (t["timestamp"] <= end)]
    h = h.loc[(h["timestamp"] >= start) & (h["timestamp"] <= end)]
    merged = t[["timestamp", RAIN_COL]].merge(
        h[["timestamp", RAIN_COL]],
        on="timestamp",
        suffixes=("_truth", "_fc"),
    )
    if merged.empty:
        return None
    merged["date"] = pd.to_datetime(merged["timestamp"], utc=True).dt.floor("D")
    daily = merged.groupby("date", as_index=False).agg(
        truth=(f"{RAIN_COL}_truth", "sum"),
        fc=(f"{RAIN_COL}_fc", "sum"),
    )
    daily = daily[(daily["truth"] >= 0) & (daily["fc"] > 0.05)]
    if len(daily) < 3:
        return None
    denom = float(daily["fc"].mean())
    if denom <= 0.05:
        return None
    scale = float(daily["truth"].mean() / denom)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return float(np.clip(scale, RAIN_SCALE_MIN, RAIN_SCALE_MAX))


def apply_rain_scale(rain: pd.Series, scale: float) -> pd.Series:
    """Scale wet-hour amounts only; leave dry hours dry."""
    values = pd.to_numeric(rain, errors="coerce").clip(lower=0.0)
    wet = values >= WET_THRESHOLD_MM
    out = values.copy()
    out.loc[wet] = values.loc[wet] * float(scale)
    return out.clip(lower=0.0)


def apply_bias_correction(
    ensemble: pd.DataFrame,
    truth: pd.DataFrame,
    window_days: int = 14,
    truth_col_map: dict[str, str] | None = None,
    hindcast: pd.DataFrame | None = None,
    rain_truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Correct live ensemble using recent truth vs hindcast.

    Temperature / humidity / pressure / wind: add mean hourly bias.
    Rain: multiply wet-hour amounts by a daily-total scale. Dry hours stay dry.
    Pass `rain_truth` when rain should use a different series than `truth`
    (Open-Meteo or a site gauge — never a wetter, different-scale product).
    """
    del truth_col_map  # kept for call-site compatibility
    if ensemble.empty or truth.empty:
        return ensemble

    out = ensemble.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    # Prefer an explicit past model series; otherwise try overlapping ensemble hours.
    model_history = hindcast if hindcast is not None and not hindcast.empty else ensemble
    end_at = out["timestamp"].min() - pd.Timedelta(hours=1)
    end_at = end_at if pd.notna(end_at) else None

    biases = estimate_channel_biases(
        truth,
        model_history,
        window_days=window_days,
        end_at=end_at,
        channels=SCALAR_COLS,
    )
    if not biases:
        biases = estimate_channel_biases(
            truth, model_history, window_days=window_days, channels=SCALAR_COLS
        )

    rain_src = rain_truth if rain_truth is not None and not rain_truth.empty else truth
    rain_scale = estimate_rain_scale(
        rain_src, model_history, window_days=window_days, end_at=end_at
    )
    if rain_scale is None:
        rain_scale = estimate_rain_scale(rain_src, model_history, window_days=window_days)

    applied = False
    for col, bias in biases.items():
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce") + bias
        out[f"{col}_bias_applied"] = bias
        applied = True

    if RAIN_COL in out.columns:
        out[RAIN_COL] = pd.to_numeric(out[RAIN_COL], errors="coerce").clip(lower=0.0)
        if rain_scale is not None:
            out[RAIN_COL] = apply_rain_scale(out[RAIN_COL], rain_scale)
            out["rainfall_mm_bias_applied"] = 0.0
            out["rainfall_mm_scale_applied"] = float(rain_scale)
            applied = True
        else:
            out["rainfall_mm_bias_applied"] = 0.0
            out["rainfall_mm_scale_applied"] = 1.0

    out["bias_corrected"] = bool(applied)
    return out


def bias_correct_series(
    forecast: pd.Series,
    truth: pd.Series,
    train_index: pd.DatetimeIndex,
) -> pd.Series:
    """Apply mean train-window bias (truth - forecast) to a hold-out forecast series."""
    train_fc = pd.to_numeric(forecast.reindex(train_index), errors="coerce")
    train_truth = pd.to_numeric(truth.reindex(train_index), errors="coerce")
    delta = (train_truth - train_fc).replace([np.inf, -np.inf], np.nan).dropna()
    if delta.empty:
        return forecast.copy()
    bias = float(delta.mean())
    corrected = pd.to_numeric(forecast, errors="coerce") + bias
    return corrected.clip(lower=0.0)
