"""Build exogenous NWP member features for local rain timing models."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import load_config

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Open-Meteo historical-forecast model ids used as ECMWF / GFS / ICON proxies.
DEFAULT_HIST_MODELS = (
    ("ecmwf", "ecmwf_ifs025"),
    ("gfs", "gfs_seamless"),
    ("icon", "icon_seamless"),
)

HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
)


def _cache_dir(cfg: dict) -> Path:
    path = Path(cfg["open_meteo"]["cache_dir"]) / "nwp_historical"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_historical_member(
    model_api: str,
    member_name: str,
    start: date,
    end: date,
    cfg: dict | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch one model series from the Open-Meteo Historical Forecast API."""
    cfg = cfg or load_config()
    station = cfg["station"]
    cache = _cache_dir(cfg) / (
        f"{member_name}_{float(station['latitude']):.4f}_{float(station['longitude']):.4f}_"
        f"{start.isoformat()}_{end.isoformat()}.csv"
    )
    if use_cache and cache.exists():
        frame = pd.read_csv(cache, parse_dates=["timestamp"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    params = {
        "latitude": float(station["latitude"]),
        "longitude": float(station["longitude"]),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "models": model_api,
        "wind_speed_unit": "ms",
    }
    response = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=90)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    frame = pd.DataFrame(hourly)
    frame["timestamp"] = pd.to_datetime(frame.pop("time"), utc=True)
    frame = frame.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "surface_pressure": "pressure_hpa",
            "precipitation": "rainfall_mm",
            "wind_speed_10m": "wind_speed_ms",
        }
    )
    frame["member"] = member_name
    frame["fetched_at"] = datetime.now(timezone.utc)
    frame.to_csv(cache, index=False)
    return frame


def fetch_historical_members(
    start: date,
    end: date,
    cfg: dict | None = None,
    models: tuple[tuple[str, str], ...] = DEFAULT_HIST_MODELS,
) -> pd.DataFrame:
    """Return long member table covering [start, end] for rain timing features."""
    cfg = cfg or load_config()
    frames: list[pd.DataFrame] = []
    for member_name, model_api in models:
        try:
            frames.append(
                fetch_historical_member(model_api, member_name, start, end, cfg=cfg)
            )
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def members_to_exogenous(
    members: pd.DataFrame,
    wet_threshold_mm: float = 0.1,
) -> pd.DataFrame:
    """
    Pivot long NWP member rows into wide hourly features.

    Adds per-member rain/humidity/temp plus agreement stats used by the
    wet/dry classifier: mean rain, rain spread, and fraction of members wet.
    """
    if members is None or members.empty:
        return pd.DataFrame()

    frame = members.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["member"] = frame["member"].astype(str)
    value_cols = [
        col
        for col in (
            "rainfall_mm",
            "humidity_pct",
            "temperature_c",
            "pressure_hpa",
            "wind_speed_ms",
        )
        if col in frame.columns
    ]
    if not value_cols or "rainfall_mm" not in value_cols:
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    for col in value_cols:
        wide = frame.pivot_table(
            index="timestamp",
            columns="member",
            values=col,
            aggfunc="first",
        )
        wide = wide.apply(pd.to_numeric, errors="coerce")
        wide = wide.mask(wide <= -900.0)
        wide.columns = [f"nwp_{str(name)}_{col}" for name in wide.columns]
        pieces.append(wide)

    out = pd.concat(pieces, axis=1).sort_index()
    rain_cols = [col for col in out.columns if col.endswith("_rainfall_mm")]
    if rain_cols:
        rain = out[rain_cols]
        out["nwp_rain_mean"] = rain.mean(axis=1, skipna=True)
        out["nwp_rain_median"] = rain.median(axis=1, skipna=True)
        out["nwp_rain_spread"] = rain.std(axis=1, skipna=True)
        out["nwp_wet_fraction"] = (rain >= wet_threshold_mm).mean(axis=1)
        out["nwp_rain_max"] = rain.max(axis=1, skipna=True)
        out["nwp_members"] = rain.notna().sum(axis=1).astype(float)
    out = out.reset_index()
    return out


def merge_weather_and_nwp(
    weather: pd.DataFrame | None,
    nwp_exo: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Left-merge weather context with NWP member features on timestamp."""
    if (weather is None or weather.empty) and (nwp_exo is None or nwp_exo.empty):
        return None
    if weather is None or weather.empty:
        return nwp_exo.copy()
    if nwp_exo is None or nwp_exo.empty:
        return weather.copy()
    left = weather.copy()
    right = nwp_exo.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    merged = left.merge(right, on="timestamp", how="left")
    return merged
