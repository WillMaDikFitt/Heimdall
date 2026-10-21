"""NOAA GFS point forecast via Open-Meteo GFS API (no login)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import load_config

GFS_URL = "https://api.open-meteo.com/v1/gfs"


def fetch_gfs(cfg: dict | None = None, forecast_days: int = 7, timeout: int = 30) -> pd.DataFrame:
    cfg = cfg or load_config()
    station = cfg["station"]
    params = {
        "latitude": float(station["latitude"]),
        "longitude": float(station["longitude"]),
        "hourly": (
            "temperature_2m,relative_humidity_2m,surface_pressure,"
            "precipitation,wind_speed_10m,wind_direction_10m"
        ),
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
    }
    response = requests.get(GFS_URL, params=params, timeout=timeout)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "surface_pressure": "pressure_hpa",
            "precipitation": "rainfall_mm",
            "wind_speed_10m": "wind_speed_ms",
            "wind_direction_10m": "wind_direction_deg",
        }
    )
    df["member"] = "gfs"
    df["fetched_at"] = datetime.now(timezone.utc)
    return df
