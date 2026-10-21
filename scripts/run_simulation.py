#!/usr/bin/env python
"""Fetch Open-Meteo reference data and write a simulated sensor stream to SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.hub import projects, runtime_config
from src.ingestion.store import init_db, replace_payloads, replace_reference
from src.reference.open_meteo import fetch_historical
from src.simulator import simulate_from_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="indra", help="Hub project id (indra, issa, all)")
    args = parser.parse_args()
    cfg = load_config()
    init_db(cfg)
    names = list(projects(cfg)) if args.project == "all" else [args.project]
    for project_id in names:
        runtime = runtime_config(cfg, project_id)
        station = runtime["station"]
        print(
            f"Fetching Open-Meteo history for {projects(cfg)[project_id]['label']} "
            f"at {station['latitude']}, {station['longitude']} ..."
        )
        reference = fetch_historical(
            latitude=float(station["latitude"]),
            longitude=float(station["longitude"]),
            cfg=runtime,
        )
        n_ref = replace_reference(reference, cfg=cfg, station_id=str(station["id"]))
        print(f"Stored {n_ref} hourly reference rows")
        payloads = simulate_from_reference(reference, cfg=runtime)
        n_sim = replace_payloads(payloads, cfg=cfg, station_id=str(station["id"]))
        print(f"Stored {n_sim} simulated {cfg['sampling']['interval_minutes']}-minute readings")
        if payloads:
            print(f"Window: {payloads[0].timestamp.isoformat()} -> {payloads[-1].timestamp.isoformat()}")


if __name__ == "__main__":
    main()
