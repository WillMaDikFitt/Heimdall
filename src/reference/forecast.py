"""Operational Open-Meteo forecast for the configured station."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import load_config

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
)

DAILY_VARIABLES = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "sunrise",
    "sunset",
)


@dataclass
class ForecastBundle:
    hourly: pd.DataFrame
    daily: pd.DataFrame
    fetched_at: datetime
    source: str


def _cache_file(cfg: dict) -> Path:
    cache_dir = Path(cfg["open_meteo"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    station = cfg["station"]
    return cache_dir / (
        f"forecast_{float(station['latitude']):.4f}_"
        f"{float(station['longitude']):.4f}.json"
    )


def _parse(payload: dict, fetched_at: datetime, source: str) -> ForecastBundle:
    hourly_data = payload["hourly"]
    hourly = pd.DataFrame(hourly_data)
    hourly["timestamp"] = pd.to_datetime(hourly.pop("time"), utc=True)
    hourly = hourly.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "surface_pressure": "pressure_hpa",
            "precipitation_probability": "rain_probability_pct",
            "precipitation": "rainfall_mm",
            "wind_speed_10m": "wind_speed_ms",
            "wind_direction_10m": "wind_direction_deg",
        }
    )

    daily_data = payload["daily"]
    daily = pd.DataFrame(daily_data)
    daily["date"] = pd.to_datetime(daily.pop("time")).dt.date
    daily = daily.rename(
        columns={
            "weather_code": "weather_code",
            "temperature_2m_max": "temperature_max_c",
            "temperature_2m_min": "temperature_min_c",
            "precipitation_sum": "rainfall_sum_mm",
            "precipitation_probability_max": "rain_probability_max_pct",
            "wind_speed_10m_max": "wind_speed_max_ms",
        }
    )
    return ForecastBundle(hourly, daily, fetched_at, source)


def fetch_forecast(
    cfg: dict | None = None,
    forecast_days: int = 7,
    cache_minutes: int = 30,
    force_refresh: bool = False,
) -> ForecastBundle:
    """Fetch a UTC hourly + local-date daily forecast, falling back to stale cache."""
    cfg = cfg or load_config()
    station = cfg["station"]
    cache_file = _cache_file(cfg)
    now = datetime.now(timezone.utc)

    cached: dict | None = None
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (
                not force_refresh
                and now - fetched_at < timedelta(minutes=cache_minutes)
            ):
                return _parse(cached["payload"], fetched_at, "Open-Meteo · cached")
        except (OSError, ValueError, KeyError, TypeError):
            cached = None

    params = {
        "latitude": float(station["latitude"]),
        "longitude": float(station["longitude"]),
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload or "daily" not in payload:
            raise RuntimeError(f"Unexpected forecast response: {payload}")
        cached_doc = {"fetched_at": now.isoformat(), "payload": payload}
        cache_file.write_text(json.dumps(cached_doc), encoding="utf-8")
        return _parse(payload, now, "Open-Meteo · live")
    except (requests.RequestException, RuntimeError, OSError):
        if cached is None:
            raise
        fetched_at = datetime.fromisoformat(cached["fetched_at"])
        return _parse(cached["payload"], fetched_at, "Open-Meteo · stale cache")


WEATHER_CODES = {
    0: ("Clear", "Clear sky"),
    1: ("Mostly clear", "Mainly clear"),
    2: ("Partly cloudy", "Variable cloud"),
    3: ("Overcast", "Overcast"),
    45: ("Fog", "Reduced visibility"),
    48: ("Rime fog", "Freezing fog"),
    51: ("Light drizzle", "Light drizzle"),
    53: ("Drizzle", "Moderate drizzle"),
    55: ("Heavy drizzle", "Dense drizzle"),
    61: ("Light rain", "Light rain"),
    63: ("Rain", "Moderate rain"),
    65: ("Heavy rain", "Heavy rain"),
    80: ("Rain showers", "Light showers"),
    81: ("Rain showers", "Moderate showers"),
    82: ("Heavy showers", "Violent showers"),
    95: ("Thunderstorm", "Thunderstorm"),
    96: ("Storm + hail", "Thunderstorm with hail"),
    99: ("Severe storm", "Thunderstorm with heavy hail"),
}


def weather_label(code: float | int | None) -> tuple[str, str]:
    if code is None or pd.isna(code):
        return "Unknown", "Forecast unavailable"
    return WEATHER_CODES.get(int(code), ("Mixed weather", f"Weather code {int(code)}"))
