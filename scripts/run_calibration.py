#!/usr/bin/env python
"""Fit per-channel offset/scale and print MAE before vs after calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.hub import projects
from src.pipeline import calibrate_project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="indra", help="Hub project id (indra, issa)")
    args = parser.parse_args()
    cfg = load_config()
    if args.project not in projects(cfg):
        raise SystemExit(f"Unknown project {args.project}")
    result = calibrate_project(args.project, cfg=cfg, simulate_if_empty=False)
    print(f"{'channel':<22} {'n':>5} {'scale':>8} {'offset':>10} {'MAE raw':>10} {'MAE fit':>10} {'MAE hold':>10}")
    print("-" * 100)
    for row in result["coeffs"]:
        hold = row.get("mae_holdout")
        hold_s = f"{hold:>10.3f}" if hold is not None else f"{'—':>10}"
        print(
            f"{row['channel']:<22} {row['n_points']:>5} {row['scale']:>8.4f} "
            f"{row['offset']:>10.4f} {row['mae_raw']:>10.3f} {row['mae']:>10.3f} {hold_s}"
        )
    print(f"\nWrote {result['n_calibrated']} calibrated rows for {result['station_id']}")


if __name__ == "__main__":
    main()
