#!/usr/bin/env python
"""Seed Heimdall with simulated weather + domain telemetry for INDRA and ISSA."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.hub import projects, runtime_config
from src.ingestion.store import init_db
from src.pipeline import calibrate_project
from src.telemetry.simulate import simulate_project_telemetry
from src.telemetry.store import replace_telemetry_domain


def seed_project(project_id: str) -> None:
    cfg = load_config()
    runtime = runtime_config(cfg, project_id)
    project = projects(cfg)[project_id]
    station_id = str(project["station"]["id"])
    label = project["label"]
    print(f">>> {label} ({station_id})", flush=True)

    print("  weather history + simulated station + calibration...", flush=True)
    try:
        result = calibrate_project(project_id, cfg=cfg, simulate_if_empty=True)
        print(
            f"  reference={result['n_reference']} sensor={result['n_sensor']} "
            f"calibrated={result['n_calibrated']}"
            + (" (practice weather)" if result.get("simulated") else ""),
            flush=True,
        )
    except Exception as exc:
        print(f"  weather/calibration skipped: {exc}", flush=True)

    print("  simulating domain telemetry...", flush=True)
    packs = simulate_project_telemetry(project, cfg=runtime, hours=24 * 14)
    for domain, rows in packs.items():
        n = replace_telemetry_domain(rows, station_id=station_id, domain=domain, cfg=cfg)
        print(f"  {domain}: {n} points", flush=True)


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    ids = [pid for pid in projects(cfg) if pid in {"indra", "issa"}]
    if not ids:
        ids = list(projects(cfg).keys())
    for project_id in ids:
        seed_project(project_id)
    print("Heimdall simulation seed complete.", flush=True)


if __name__ == "__main__":
    main()
