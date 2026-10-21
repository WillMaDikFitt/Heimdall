#!/usr/bin/env python
"""Fetch NASA POWER hourly reference and store in SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingestion.store import init_db, replace_power_readings
from src.reference.nasa_power import fetch_power_hourly


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    days = int(cfg.get("validation", {}).get("nasa_power_days", 90))
    print(f"Fetching NASA POWER hourly data ({days} days)...")
    df = fetch_power_hourly(cfg=cfg, days=min(days, 90))  # API limit ~90d per call
    n = replace_power_readings(df, cfg=cfg)
    print(f"Stored {n} NASA POWER hourly rows in {cfg['storage']['sqlite_path']}")


if __name__ == "__main__":
    main()
