#!/usr/bin/env python
"""Snapshot issued forecasts and score against archive truth when available."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ingestion.store import init_db, replace_forecast_verification_scores
from src.reference.forecast_verify import (
    snapshot_forecast,
    snapshot_stored_ensemble,
    verify_archived_forecasts,
)


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    try:
        path = snapshot_stored_ensemble(cfg=cfg)
        print(f"Archived Overview ensemble: {path.name}")
    except Exception as exc:
        print(f"Could not snapshot ensemble: {exc}")
        try:
            path = snapshot_forecast(cfg=cfg)
            print(f"Fell back to Open-Meteo snapshot: {path.name}")
        except Exception as exc2:
            print(f"Could not snapshot forecast (network): {exc2}")

    scores = verify_archived_forecasts(cfg=cfg)
    replace_forecast_verification_scores(scores, cfg=cfg)
    if not scores:
        print("No mature archived forecasts to verify yet (need 48h+ lag).")
        return

    print(f"{'issued':<22} {'channel':<18} {'model':<24} {'MAE':>8} {'dayMAE':>8} {'CSI':>6}")
    print("-" * 92)
    for row in scores:
        day = row.get("daily_mae")
        csi = row.get("csi")
        day_s = f"{day:>8.2f}" if day is not None else f"{'—':>8}"
        csi_s = f"{csi:>6.3f}" if csi is not None else f"{'—':>6}"
        print(
            f"{row['issued_at'][:19]:<22} {row['channel']:<18} {row['model']:<24} "
            f"{row['mae']:>8.3f} {day_s} {csi_s}"
        )
    print(f"\nSaved {len(scores)} verification score rows.")


if __name__ == "__main__":
    main()
