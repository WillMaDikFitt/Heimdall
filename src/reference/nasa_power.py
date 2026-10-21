"""NASA POWER hourly reference for independent validation (no API key)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import load_config
from src.schema import CHANNEL_NAMES

POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

POWER_PARAMS = {
    "T2M": "temperature_c",
    "RH2M": "humidity_pct",
    "PS": "pressure_hpa",
    "WS10M": "wind_speed_ms",
    "WD10M": "wind_direction_deg",
    "PRECTOTCORR": "rainfall_mm",
}


def _cache_path(cfg: dict, start: str, end: str) -> Path:
    cache_dir = Path(cfg["open_meteo"]["cache_dir"]) / "nasa_power"
    cache_dir.mkdir(parents=True, exist_ok=True)
    station = cfg["station"]
    return cache_dir / (
        f"power_{float(station['latitude']):.4f}_{float(station['longitude']):.4f}_"
        f"{start}_{end}.json"
    )


def fetch_power_hourly(
    cfg: dict | None = None,
    days: int | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download NASA POWER hourly parameters for the configured station.
    PS is returned in kPa and converted to hPa.
    """
    cfg = cfg or load_config()
    station = cfg["station"]
    days = int(days or cfg.get("validation", {}).get("nasa_power_days", 365))
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    start_s, end_s = start.isoformat(), end.isoformat()
    cache_file = _cache_path(cfg, start_s, end_s)

    if cache_file.exists() and not force_refresh:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        params = {
            "parameters": ",".join(POWER_PARAMS.keys()),
            "community": "RE",
            "longitude": float(station["longitude"]),
            "latitude": float(station["latitude"]),
            "start": start_s.replace("-", ""),
            "end": end_s.replace("-", ""),
            "format": "JSON",
        }
        response = requests.get(POWER_HOURLY_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

    props = payload.get("properties", {}).get("parameter", {})
    if not props:
        raise RuntimeError("NASA POWER response missing parameter block")

    # POWER returns YYYYMMDDHH keys per parameter
    series: dict[str, pd.Series] = {}
    for api_key, col in POWER_PARAMS.items():
        raw = props.get(api_key, {})
        if not raw:
            continue
        idx = pd.to_datetime(list(raw.keys()), format="%Y%m%d%H", utc=True)
        values = pd.to_numeric(list(raw.values()), errors="coerce")
        series[col] = pd.Series(values, index=idx)

    if not series:
        raise RuntimeError("No NASA POWER variables parsed")

    df = pd.DataFrame(series).sort_index()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    # POWER uses large negative fill values for missing hours.
    for col in df.columns:
        if col == "timestamp":
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        df[col] = values.mask(values <= -900.0)
    if "pressure_hpa" in df.columns:
        # PS in kPa -> hPa
        df["pressure_hpa"] = df["pressure_hpa"] * 10.0
    for col in CHANNEL_NAMES:
        if col not in df.columns:
            df[col] = float("nan")
    return df[["timestamp", *CHANNEL_NAMES]]
