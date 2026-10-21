"""Met Norway Locationforecast 2.0 (no API key; requires User-Agent)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import load_config

METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
USER_AGENT = "MechatronWeatherStation/1.0 github.com/mechatron-weather"


def fetch_met_norway(cfg: dict | None = None, timeout: int = 30) -> pd.DataFrame:
    cfg = cfg or load_config()
    station = cfg["station"]
    headers = {"User-Agent": USER_AGENT}
    params = {"lat": float(station["latitude"]), "lon": float(station["longitude"])}
    response = requests.get(METNO_URL, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    timeseries = payload["properties"]["timeseries"]
    rows: list[dict] = []
    for entry in timeseries:
        ts = pd.to_datetime(entry["time"], utc=True)
        details = entry["data"]["instant"]["details"]
        next1 = entry["data"].get("next_1_hours", {})
        next6 = entry["data"].get("next_6_hours", {})
        next12 = entry["data"].get("next_12_hours", {})
        precip = next1.get("details", {}).get("precipitation_amount")
        if precip is None:
            precip = next6.get("details", {}).get("precipitation_amount")
        if precip is None:
            precip = next12.get("details", {}).get("precipitation_amount")
        rows.append(
            {
                "timestamp": ts,
                "temperature_c": details.get("air_temperature"),
                "humidity_pct": details.get("relative_humidity"),
                "pressure_hpa": details.get("air_pressure_at_sea_level"),
                "wind_speed_ms": details.get("wind_speed"),
                "wind_direction_deg": details.get("wind_from_direction"),
                "rainfall_mm": precip or 0.0,
                "member": "met_norway",
                "fetched_at": datetime.now(timezone.utc),
            }
        )
    return pd.DataFrame(rows)
