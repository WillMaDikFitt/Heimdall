#!/usr/bin/env python
"""
Retrain local gradient-boosting models using non-circular truth (NASA POWER when available)
and Open-Meteo as exogenous NWP features. Includes walk-forward and rain-aware metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess


def main() -> None:
    steps = [
        ["python", "scripts/run_fetch_power.py"],
        ["python", "scripts/run_forecast_backtest.py"],
        ["python", "scripts/run_walkforward_backtest.py"],
    ]
    for cmd in steps:
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)
    print("\nLocal model retrain complete. Check walkforward_scores and forecast_scores in SQLite.")


if __name__ == "__main__":
    main()
