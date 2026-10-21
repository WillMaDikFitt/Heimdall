#!/usr/bin/env python
"""
Historical forecast backtest.

Example (your idea): take the last 6 days of Open-Meteo history at Leshka,
train on the first 3 days, predict the next 3 days, then score against the
held-out actuals we already have.

Also optionally trains a second model on *all* history before the last 3 days
so you can see how much more data helps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config
from src.forecast.evaluate import run_channel_backtest
from src.forecast.nwp_features import (
    fetch_historical_members,
    members_to_exogenous,
    merge_weather_and_nwp,
)
from src.forecast.walkforward import run_walkforward_backtest
from src.ingestion.store import (
    init_db,
    load_nwp_members,
    load_power_readings,
    load_reference,
    replace_forecast_results,
    replace_walkforward_scores,
)


def _usable_hours(frame: pd.DataFrame, channel: str = "temperature_c") -> int:
    if frame.empty or channel not in frame.columns:
        return 0
    values = pd.to_numeric(frame[channel], errors="coerce")
    return int(values.mask(values <= -900.0).notna().sum())


def _rain_nwp_exogenous(reference: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame | None, str]:
    """Weather context plus ECMWF/GFS/ICON historical-forecast member features."""
    weather_cols = [
        name
        for name in ("temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms")
        if name in reference.columns
    ]
    weather = reference[["timestamp", *weather_cols]].copy() if weather_cols else None

    ts = pd.to_datetime(reference["timestamp"], utc=True)
    start = ts.min().date()
    end = ts.max().date()
    print(f"Fetching historical NWP members for rain timing ({start} -> {end})...")
    hist_members = fetch_historical_members(start, end, cfg=cfg)
    live_members = load_nwp_members(cfg=cfg)
    members = (
        pd.concat([hist_members, live_members], ignore_index=True)
        if not hist_members.empty or not live_members.empty
        else pd.DataFrame()
    )
    nwp_exo = members_to_exogenous(members) if not members.empty else pd.DataFrame()
    if nwp_exo.empty:
        note = " [truth=Open-Meteo · NWP features unavailable]"
        return weather, note

    n_feat = len([c for c in nwp_exo.columns if str(c).startswith("nwp_")])
    covered = int(nwp_exo["nwp_rain_mean"].notna().sum()) if "nwp_rain_mean" in nwp_exo.columns else 0
    print(f"NWP rain features ready: {n_feat} columns, {covered} hours with member rain.")
    merged = merge_weather_and_nwp(weather, nwp_exo)
    note = " [truth=Open-Meteo · NWP members on]"
    return merged, note


def main() -> None:
    cfg = load_config()
    init_db(cfg)
    station_id = cfg["station"]["id"]
    reference = load_reference(cfg=cfg, station_id=station_id)
    power = load_power_readings(cfg=cfg)

    power_hours = _usable_hours(power)
    ref_hours = _usable_hours(reference)
    # Prefer NASA POWER when it has enough real (non-fill) hours; otherwise use Open-Meteo.
    if power_hours >= max(168, ref_hours // 2):
        truth, truth_label = power, "NASA POWER"
    elif not reference.empty:
        truth, truth_label = reference, "Open-Meteo reference"
    else:
        truth, truth_label = power, "NASA POWER"
    if truth.empty:
        raise SystemExit("No reference data. Run simulation or fetch NASA POWER first.")
    exogenous = reference if not reference.empty and truth_label != "Open-Meteo reference" else None
    print(f"Usable hours — NASA POWER: {power_hours}, Open-Meteo: {ref_hours}")

    rain_exo: pd.DataFrame | None = None
    rain_note = ""
    if not reference.empty:
        rain_exo, rain_note = _rain_nwp_exogenous(reference, cfg)

    fc = cfg.get("forecast", {})
    train_days = int(fc.get("train_days", 3))
    forecast_days = int(fc.get("forecast_days", 3))
    channels = list(fc.get("channels", ["temperature_c"]))
    also_full = bool(fc.get("also_full_history", True))

    all_preds: list[pd.DataFrame] = []
    all_scores: list[dict] = []
    wf_scores: list[dict] = []

    print(f"Default truth source: {truth_label} ({len(truth)} hourly rows)")
    print(
        "Rainfall amount is scored on Open-Meteo when available "
        "(live outlook scale). Other channels use the default truth source."
    )

    modes = [(False, f"last_{train_days}d train")]
    if also_full:
        modes.append((True, "full history train"))

    for use_full, label in modes:
        print(f"\n=== Backtest mode: {label} | forecast {forecast_days}d ===")
        for channel in channels:
            if channel == "rainfall_mm" and not reference.empty:
                channel_truth = reference
                channel_exo = rain_exo
                channel_note = rain_note
            else:
                channel_truth = truth
                channel_exo = exogenous
                channel_note = ""

            preds, scores = run_channel_backtest(
                channel_truth,
                value_col=channel,
                channel=channel,
                train_days=train_days,
                forecast_days=forecast_days,
                use_full_history_train=use_full,
                exogenous=channel_exo,
            )
            suffix = "_fullhist" if use_full else ""
            preds = preds.copy()
            preds["model"] = preds["model"] + suffix
            for row in scores:
                row["model"] = row["model"] + suffix
            all_preds.append(preds)
            all_scores.extend(scores)

            print(f"\n{channel}{channel_note}")
            if channel == "rainfall_mm":
                print(
                    f"{'model':<28} {'dayMAE':>7} {'CSI':>6} {'CSI6':>6} {'CSI24':>6} {'hit%':>5}"
                )
                print("-" * 68)
                for row in scores:
                    day = row.get("daily_mae")
                    csi = row.get("csi")
                    c6 = row.get("csi_6h")
                    c24 = row.get("csi_24h")
                    hit = row.get("hit_rate")
                    day_s = f"{day:.1f}" if day is not None and pd.notna(day) else "—"
                    csi_s = f"{csi:.3f}" if csi is not None and pd.notna(csi) else "—"
                    c6_s = f"{c6:.3f}" if c6 is not None and pd.notna(c6) else "—"
                    c24_s = f"{c24:.3f}" if c24 is not None and pd.notna(c24) else "—"
                    hit_s = f"{100 * hit:.0f}" if hit is not None and pd.notna(hit) else "—"
                    print(
                        f"{row['model']:<28} {day_s:>7} {csi_s:>6} {c6_s:>6} {c24_s:>6} {hit_s:>5}"
                    )
            else:
                print(f"{'model':<28} {'MAE':>8} {'RMSE':>8} {'bias':>8} {'n':>5}")
                print("-" * 62)
                for row in scores:
                    print(
                        f"{row['model']:<28} {row['mae']:>8.3f} {row['rmse']:>8.3f} "
                        f"{row['bias']:>8.3f} {row['n']:>5}"
                    )

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    replace_forecast_results(combined, all_scores, cfg=cfg)

    horizons = tuple(int(h) for h in fc.get("horizons", [6, 24, 72]))
    step_hours = int(fc.get("walkforward_step_hours", 24))
    for channel in channels:
        wf_truth = reference if channel == "rainfall_mm" and not reference.empty else truth
        if channel not in wf_truth.columns:
            continue
        try:
            _, wf = run_walkforward_backtest(
                wf_truth,
                value_col=channel,
                channel=channel,
                horizons=horizons,
                step_hours=step_hours,
            )
            wf_scores.extend(wf)
        except ValueError:
            continue
    replace_walkforward_scores(wf_scores, cfg=cfg)

    print(f"\nSaved {len(all_scores)} hold-out score rows and {len(combined)} prediction points.")
    print(f"Saved {len(wf_scores)} walk-forward score rows.")


if __name__ == "__main__":
    main()
