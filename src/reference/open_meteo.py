"""Open-Meteo historical weather client with on-disk CSV cache."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import load_config
from src.schema import CHANNEL_NAMES

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
)

REF_TO_CHANNEL = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "surface_pressure": "pressure_hpa",
    "precipitation": "rainfall_mm",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
}


def default_date_range(cfg: dict | None = None) -> tuple[date, date]:
    cfg = cfg or load_config()
    lag = int(cfg["open_meteo"]["archive_lag_days"])
    days = int(cfg["open_meteo"]["historical_days"])
    end = datetime.now(timezone.utc).date() - timedelta(days=lag)
    start = end - timedelta(days=days - 1)
    return start, end


def _cache_path(cache_dir: Path, lat: float, lon: float, start: date, end: date) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"openmeteo_{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.csv"
    return cache_dir / name


def fetch_historical(
    latitude: float | None = None,
    longitude: float | None = None,
    start: date | None = None,
    end: date | None = None,
    cfg: dict | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return hourly reference weather in station channel names, UTC index."""
    cfg = cfg or load_config()
    lat = latitude if latitude is not None else float(cfg["station"]["latitude"])
    lon = longitude if longitude is not None else float(cfg["station"]["longitude"])
    if start is None or end is None:
        default_start, default_end = default_date_range(cfg)
        start = start or default_start
        end = end or default_end

    cache_dir = Path(cfg["open_meteo"]["cache_dir"])
    cache_file = _cache_path(cache_dir, lat, lon, start, end)
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError(f"Unexpected Open-Meteo response: {payload}")

    hourly = payload["hourly"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
    for api_name, channel in REF_TO_CHANNEL.items():
        frame[channel] = hourly.get(api_name)

    for channel in CHANNEL_NAMES:
        if channel not in frame.columns:
            frame[channel] = pd.NA

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if use_cache:
        frame.to_csv(cache_file, index=False)
    return frame
