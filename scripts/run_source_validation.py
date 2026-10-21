#!/usr/bin/env python
"""Compare Open-Meteo, NASA POWER, IMD, and calibrated station readings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config
from src.ingestion.store import (
    init_db,
    load_calibrated,
    load_imd_daily_rain,
    load_power_readings,
    load_reference,
    replace_source_validation_scores,
)
from src.validation.compare import compare_daily_rain, compare_hourly_sources


def _calibrated_wide(calibrated: pd.DataFrame) -> pd.DataFrame:
    if calibrated.empty:
        return calibrated
    wide = calibrated.pivot_table(
        index="timestamp", columns="channel", values="calibrated", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    station_id = cfg["station"]["id"]
    reference = load_reference(cfg=cfg, station_id=station_id)
    power = load_power_readings(cfg=cfg)
    imd = load_imd_daily_rain(cfg=cfg)
    calibrated = _calibrated_wide(load_calibrated(cfg=cfg, station_id=station_id))

    scores: list[dict] = []

    if not reference.empty and not power.empty:
        scores.extend(compare_hourly_sources(reference, power, "Open-Meteo", "NASA POWER"))
    if not reference.empty and not imd.empty:
        rain_score = compare_daily_rain(reference, imd, "Open-Meteo", "IMD grid")
        if rain_score:
            scores.append(rain_score)
    if not calibrated.empty and not power.empty:
        scores.extend(compare_hourly_sources(calibrated, power, "Calibrated station", "NASA POWER"))
    if not calibrated.empty and not imd.empty:
        rain_score = compare_daily_rain(calibrated, imd, "Calibrated station", "IMD grid")
        if rain_score:
            scores.append(rain_score)

    replace_source_validation_scores(scores, cfg=cfg)

    if not scores:
        print("No overlapping data for validation. Run simulation + fetch_power first.")
        return

    print(f"{'A':<20} {'B':<20} {'channel':<18} {'gran':<8} {'MAE':>8} {'bias':>8} {'n':>5}")
    print("-" * 95)
    for row in scores:
        print(
            f"{row['source_a']:<20} {row['source_b']:<20} {row['channel']:<18} "
            f"{row['granularity']:<8} {row['mae']:>8.3f} {row['bias']:>8.3f} {row['n']:>5}"
        )
    print(f"\nSaved {len(scores)} validation score rows.")


if __name__ == "__main__":
    main()
