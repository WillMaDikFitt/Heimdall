#!/usr/bin/env python
"""Fetch Open-Meteo weather and fit calibration for every hub project pin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.hub import projects
from src.pipeline import calibrate_hub, calibrate_project


def _print_result(result: dict) -> None:
    print(
        f"\n{result['label']} ({result['station_id']}) "
        f"{result['latitude']:.5f}°N, {result['longitude']:.5f}°E "
        f"· {result['area_radius_km']:.0f} km area"
    )
    print(
        f"  reference hours: {result['n_reference']} · "
        f"station rows: {result['n_sensor']} · "
        f"calibrated rows: {result['n_calibrated']}"
        + (" · practice sensor data" if result["simulated"] else "")
    )
    print(f"  {'channel':<22} {'n':>5} {'scale':>8} {'offset':>10} {'MAE raw':>10} {'MAE fit':>10}")
    print("  " + "-" * 70)
    for row in result["coeffs"]:
        print(
            f"  {row['channel']:<22} {row['n_points']:>5} {row['scale']:>8.4f} "
            f"{row['offset']:>10.4f} {row['mae_raw']:>10.3f} {row['mae']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        choices=["all", *projects(load_config()).keys()],
        default="all",
        help="Hub project to calibrate (default: all)",
    )
    args = parser.parse_args()
    cfg = load_config()
    if args.project == "all":
        results = calibrate_hub(cfg)
    else:
        results = [calibrate_project(args.project, cfg=cfg)]
    for result in results:
        _print_result(result)


if __name__ == "__main__":
    main()
