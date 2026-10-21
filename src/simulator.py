"""Realistic cheap-sensor distortions applied to Open-Meteo reference data."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import load_config
from src.schema import CHANNEL_NAMES, Meta, Readings, SensorPayload


def _upsample_reference(reference: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    df = reference.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    freq = f"{interval_minutes}min"
    hours_per_sample = interval_minutes / 60.0

    rain = df["rainfall_mm"].resample(freq).ffill() * hours_per_sample
    wind_dir = df["wind_direction_deg"].resample(freq).ffill()
    other = df.drop(columns=["rainfall_mm", "wind_direction_deg"], errors="ignore")
    other = other.resample(freq).interpolate(method="time")

    out = other
    out["rainfall_mm"] = rain.clip(lower=0)
    out["wind_direction_deg"] = wind_dir
    return out.reset_index()


def _clip_channel(name: str, values: np.ndarray) -> np.ndarray:
    if name == "humidity_pct":
        return np.clip(values, 0, 100)
    if name in {"rainfall_mm", "wind_speed_ms"}:
        return np.clip(values, 0, None)
    if name == "wind_direction_deg":
        return np.mod(values, 360.0)
    return values


def simulate_from_reference(
    reference: pd.DataFrame,
    cfg: dict | None = None,
) -> list[SensorPayload]:
    cfg = cfg or load_config()
    sim = cfg["simulation"]
    rng = np.random.default_rng(int(sim["seed"]))
    interval = int(cfg["sampling"]["interval_minutes"])
    days = int(sim["days"])

    truth = _upsample_reference(reference, interval)
    cutoff = truth["timestamp"].max() - pd.Timedelta(days=days)
    truth = truth.loc[truth["timestamp"] >= cutoff].reset_index(drop=True)
    if truth.empty:
        raise ValueError("Reference data does not cover the requested simulation window")

    t0 = truth["timestamp"].iloc[0]
    days_elapsed = (truth["timestamp"] - t0).dt.total_seconds() / 86400.0
    n = len(truth)

    distorted: dict[str, np.ndarray] = {}
    for channel in CHANNEL_NAMES:
        base = pd.to_numeric(truth[channel], errors="coerce").to_numpy(dtype=float)
        scale = float(sim["scale"].get(channel, 1.0))
        bias = float(sim["bias"].get(channel, 0.0))
        noise_std = float(sim["noise_std"].get(channel, 0.0))
        drift = float(sim["drift_per_day"].get(channel, 0.0))
        spike_mag = float(sim["spike_magnitude"].get(channel, 0.0))

        noise = rng.normal(0.0, noise_std, size=n)
        series = scale * base + bias + noise + drift * days_elapsed.to_numpy()

        spike_mask = rng.random(n) < float(sim["spike_probability"])
        spike_sign = rng.choice(np.array([-1.0, 1.0]), size=n)
        series = np.where(spike_mask, series + spike_sign * spike_mag, series)
        distorted[channel] = _clip_channel(channel, series)

    missing_mask = rng.random(n) < float(sim["missing_probability"])
    station_id = cfg["station"]["id"]
    payloads: list[SensorPayload] = []
    for i, row in truth.iterrows():
        if missing_mask[i]:
            continue
        readings = {ch: float(distorted[ch][i]) for ch in CHANNEL_NAMES}
        if any(np.isnan(v) for v in readings.values()):
            continue
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        payloads.append(
            SensorPayload(
                station_id=station_id,
                timestamp=ts,
                readings=Readings(**readings),
                meta=Meta(firmware="sim-1.0.0", battery_v=3.85),
                source="simulator",
            )
        )
    return payloads


def payloads_to_frame(payloads: list[SensorPayload]) -> pd.DataFrame:
    return pd.DataFrame([p.to_row() for p in payloads])
