"""IMD 0.25° daily rainfall grid extraction (public binary download, no login)."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import load_config

# Official IMD yearwise binary download (same endpoint used by imdlib).
IMD_RAIN_POST_URL = "https://imdpune.gov.in/cmpg/Griddata/rainfall.php"
LAT_SIZE = 129
LON_SIZE = 135
LATS = np.linspace(6.5, 38.5, LAT_SIZE)
LONS = np.linspace(66.5, 100.0, LON_SIZE)


def _imd_cache_dir(cfg: dict) -> Path:
    d = Path(cfg["open_meteo"]["cache_dir"]) / "imd"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _nearest_cell_index(lat: float, lon: float) -> tuple[int, int]:
    lat_i = int(np.argmin(np.abs(LATS - lat)))
    lon_i = int(np.argmin(np.abs(LONS - lon)))
    return lat_i, lon_i


def _download_year_binary(year: int, cache_dir: Path) -> Path | None:
    """Download one year of IMD 0.25° rain as a .grd binary file."""
    dest = cache_dir / f"rain_{year}.grd"
    expected_days = 366 if calendar.isleap(year) else 365
    expected_floats = expected_days * LAT_SIZE * LON_SIZE
    if dest.exists() and dest.stat().st_size >= expected_floats * 4:
        return dest

    try:
        response = requests.post(IMD_RAIN_POST_URL, data={"rain": str(year)}, timeout=180)
        response.raise_for_status()
    except requests.RequestException:
        return None

    content = response.content
    if len(content) < expected_floats * 4:
        # Incomplete or HTML error page.
        return None
    dest.write_bytes(content)
    return dest


def _extract_from_grd(path: Path, year: int, lat: float, lon: float) -> pd.DataFrame:
    days = 366 if calendar.isleap(year) else 365
    nlen = days * LAT_SIZE * LON_SIZE
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < nlen:
        raise ValueError(f"IMD file too small: {path.name} has {raw.size} floats, need {nlen}")
    if raw.size > nlen:
        raw = raw[:nlen]

    # File order: (day, lat, lon) then transpose to (day, lon, lat) as in imdlib.
    data = np.transpose(raw.reshape((days, LAT_SIZE, LON_SIZE), order="C"), (0, 2, 1))
    lat_i, lon_i = _nearest_cell_index(lat, lon)
    series = data[:, lon_i, lat_i].astype(float)
    series = np.where(series < -900, np.nan, series)

    start = date(year, 1, 1)
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"date": dates.date, "rainfall_mm": series})


def fetch_imd_daily_rain(
    cfg: dict | None = None,
    years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """
    Download IMD 0.25° daily rainfall binaries and extract the nearest grid cell.

    Falls back to an empty frame if downloads fail (verification then skips IMD).
    """
    cfg = cfg or load_config()
    val = cfg.get("validation", {})
    lat = float(val.get("imd_grid_lat", 25.25))
    lon = float(val.get("imd_grid_lon", 92.25))
    cache_dir = _imd_cache_dir(cfg)
    now_year = datetime.now(timezone.utc).year
    if years is None:
        years = tuple(range(now_year - 2, now_year + 1))

    frames: list[pd.DataFrame] = []
    for year in years:
        path = _download_year_binary(year, cache_dir)
        if path is None:
            continue
        try:
            frames.append(_extract_from_grd(path, year, lat, lon))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "rainfall_mm", "grid_lat", "grid_lon"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    out["grid_lat"] = lat
    out["grid_lon"] = lon
    # Drop trailing all-NaN days (partial current year placeholders).
    out = out.dropna(subset=["rainfall_mm"])
    return out.reset_index(drop=True)
