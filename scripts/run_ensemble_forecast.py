#!/usr/bin/env python
"""Fetch NWP members, build bias-corrected ensemble, store in SQLite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config
from src.forecast.ensemble import apply_bias_correction, build_ensemble, fetch_all_members
from src.ingestion.store import (
    init_db,
    load_calibrated,
    load_power_readings,
    load_readings,
    load_reference,
    replace_ensemble_forecast,
    replace_nwp_members,
)

LogFn = Callable[[str], None]


def _calibrated_wide(cfg: dict, station_id: str) -> pd.DataFrame:
    cal = load_calibrated(cfg=cfg, station_id=station_id)
    if cal.empty:
        return cal
    wide = cal.pivot_table(
        index="timestamp", columns="channel", values="calibrated", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide


def _scalar_truth(cfg: dict) -> pd.DataFrame:
    """Temp/humidity truth: calibrated, else NASA POWER, else Open-Meteo."""
    station_id = cfg["station"]["id"]
    cal = _calibrated_wide(cfg, station_id)
    if not cal.empty:
        return cal
    power = load_power_readings(cfg=cfg)
    if not power.empty:
        return power
    return load_reference(cfg=cfg, station_id=station_id)


def _has_live_sensor(cfg: dict) -> bool:
    readings = load_readings(cfg["station"]["id"], cfg=cfg)
    return (
        not readings.empty
        and "source" in readings.columns
        and readings["source"].eq("sensor").any()
    )


def _rain_truth(cfg: dict, reference: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rain truth must match millimetre scale of the ensemble."""
    if _has_live_sensor(cfg):
        cal = _calibrated_wide(cfg, cfg["station"]["id"])
        if not cal.empty and "rainfall_mm" in cal.columns:
            return cal, "site sensor"
    if not reference.empty:
        return reference, "Open-Meteo"
    return pd.DataFrame(), "none"


def run_ensemble_for_config(cfg: dict, log: LogFn | None = None) -> int:
    """Build and store ensemble for cfg['station']. Returns hour count."""

    def _emit(message: str) -> None:
        if log:
            log(message)
        else:
            print(message, flush=True)

    init_db(cfg)
    station_id = str(cfg["station"]["id"])
    days = int(cfg["forecast"].get("operational_days", 7))
    _emit(f"Fetching NWP members ({days}-day outlook) for {station_id}...")
    members = fetch_all_members(cfg=cfg, forecast_days=days)
    if not members:
        raise RuntimeError("No NWP members returned — check network.")

    replace_nwp_members(members, cfg=cfg, station_id=station_id)
    ensemble = build_ensemble(members)
    truth = _scalar_truth(cfg)
    reference = load_reference(cfg=cfg, station_id=station_id)
    rain_truth, rain_truth_name = _rain_truth(cfg, reference)
    if not truth.empty:
        window = int(cfg["forecast"].get("bias_window_days", 14))
        hindcast = reference if not reference.empty else None
        ensemble = apply_bias_correction(
            ensemble,
            truth,
            window_days=window,
            hindcast=hindcast,
            rain_truth=rain_truth if not rain_truth.empty else None,
        )
        applied = False
        if "bias_corrected" in ensemble.columns and len(ensemble):
            applied = bool(pd.to_numeric(ensemble["bias_corrected"], errors="coerce").fillna(0).iloc[0])
        if applied:
            scale = None
            if "rainfall_mm_scale_applied" in ensemble.columns and len(ensemble):
                scale = float(ensemble["rainfall_mm_scale_applied"].iloc[0])
            rain_note = (
                f" rain scale {scale:.2f}x on wet hours (truth={rain_truth_name})"
                if scale is not None
                else ""
            )
            _emit(f"Applied {window}-day bias correction using truth vs Open-Meteo hindcast.{rain_note}")
        else:
            _emit("Bias correction skipped (not enough overlapping history).")
    else:
        _emit("No truth history available — storing raw ensemble.")

    n = replace_ensemble_forecast(ensemble, cfg=cfg, station_id=station_id)
    source = ensemble["source"].iloc[0] if n else "none"
    _emit(f"Stored ensemble ({n} hours) from: {source}")
    return n


def main() -> None:
    cfg = load_config()
    run_ensemble_for_config(cfg)


if __name__ == "__main__":
    main()
