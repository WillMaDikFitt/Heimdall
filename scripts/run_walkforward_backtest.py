#!/usr/bin/env python
"""Walk-forward backtest with lead-time breakdown and optional NWP exogenous features."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.forecast.walkforward import run_walkforward_backtest
from src.ingestion.store import init_db, load_power_readings, load_reference, replace_walkforward_scores
import pandas as pd


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    fc = cfg.get("forecast", {})
    horizons = tuple(int(h) for h in fc.get("horizons", [6, 24, 72]))
    step_hours = int(fc.get("walkforward_step_hours", 24))
    channels = list(fc.get("channels", ["temperature_c"]))

    power = load_power_readings(cfg=cfg)
    reference = load_reference(cfg=cfg, station_id=cfg["station"]["id"])

    def _usable_hours(frame: pd.DataFrame, channel: str = "temperature_c") -> int:
        if frame.empty or channel not in frame.columns:
            return 0
        values = pd.to_numeric(frame[channel], errors="coerce")
        return int(values.mask(values <= -900.0).notna().sum())

    power_hours = _usable_hours(power)
    ref_hours = _usable_hours(reference)
    if power_hours >= max(168, ref_hours // 2):
        truth, source = power, "NASA POWER"
    elif not reference.empty:
        truth, source = reference, "Open-Meteo reference"
    else:
        truth, source = power, "NASA POWER"
    if truth.empty:
        raise SystemExit("No truth series. Run simulation or fetch NASA POWER first.")

    exogenous = reference
    all_scores: list[dict] = []
    print(f"Walk-forward backtest on {source} ({len(truth)} rows)")
    print(f"Usable hours — NASA POWER: {power_hours}, Open-Meteo: {ref_hours}")
    for channel in channels:
        if channel not in truth.columns:
            continue
        _, scores = run_walkforward_backtest(
            truth,
            value_col=channel,
            channel=channel,
            horizons=horizons,
            step_hours=step_hours,
        )
        all_scores.extend(scores)
        print(f"\n{channel} @ horizons {horizons}")
        print(f"{'model':<22} {'lead':>5} {'MAE':>8} {'std':>8} {'skill':>8} {'n':>5}")
        print("-" * 62)
        for row in scores:
            skill = row.get("skill_vs_naive")
            skill_s = f"{skill:.2f}" if skill is not None else "—"
            print(
                f"{row['model']:<22} {row['lead_hours']:>5} "
                f"{row['mae']:>8.3f} {row.get('mae_std', 0):>8.3f} {skill_s:>8} {row['n']:>5}"
            )

    replace_walkforward_scores(all_scores, cfg=cfg)
    print(f"\nSaved {len(all_scores)} walk-forward score rows.")


if __name__ == "__main__":
    main()
