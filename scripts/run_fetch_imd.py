#!/usr/bin/env python
"""Fetch IMD 0.25° daily rainfall for the nearest grid cell."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingestion.store import init_db, replace_imd_daily_rain
from src.reference.imd_rain import fetch_imd_daily_rain


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    print("Fetching IMD 0.25° daily rainfall...")
    df = fetch_imd_daily_rain(cfg=cfg)
    if df.empty:
        print("IMD download failed or unavailable — check network / IMD file URLs.")
        return
    n = replace_imd_daily_rain(df, cfg=cfg)
    print(f"Stored {n} IMD daily rain rows in {cfg['storage']['sqlite_path']}")


if __name__ == "__main__":
    main()
