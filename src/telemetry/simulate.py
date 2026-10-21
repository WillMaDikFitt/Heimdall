"""Realistic simulated telemetry for INDRA hydro / ISSA soil until hardware is live."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.telemetry.domains import DOMAIN_CHANNELS, project_domains


def _timestamps(hours: int, interval_minutes: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.now(tz="UTC").floor("min")
    periods = max(2, int(hours * 60 / interval_minutes))
    return pd.date_range(end=end, periods=periods, freq=f"{interval_minutes}min", tz="UTC")


def _series(
    n: int,
    rng: np.random.Generator,
    *,
    base: float,
    amp: float,
    noise: float,
    drift: float = 0.0,
    floor: float | None = None,
    ceil: float | None = None,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(n, dtype=float)
    diurnal = amp * np.sin(2 * np.pi * t / (24 * 12) + phase)  # ~5-min samples
    values = base + diurnal + drift * (t / max(n, 1)) + rng.normal(0.0, noise, size=n)
    if floor is not None:
        values = np.maximum(values, floor)
    if ceil is not None:
        values = np.minimum(values, ceil)
    return values


def _rows_for_channel(
    timestamps: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    station_id: str,
    domain: str,
    channel: str,
    unit: str,
    source: str = "simulator",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ts, value in zip(timestamps, values):
        if np.isnan(value):
            continue
        out.append(
            {
                "station_id": station_id,
                "timestamp": ts,
                "domain": domain,
                "channel": channel,
                "value": float(value),
                "unit": unit,
                "quality_flag": "good",
                "source": source,
                "meta": {"simulated": True},
            }
        )
    return out


def simulate_indra_hydro(
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    station_id: str,
) -> dict[str, list[dict[str, Any]]]:
    n = len(timestamps)
    # Reservoir level: slow seasonal-ish drift + small waves
    level = _series(n, rng, base=412.4, amp=0.08, noise=0.015, drift=0.04, floor=410.0, ceil=416.0)
    # Velocity correlates weakly with falling level spikes (release / inflow)
    velocity = 0.35 + 0.12 * np.sin(np.linspace(0, 10, n)) + rng.normal(0, 0.03, n)
    velocity = np.clip(velocity + 0.05 * np.maximum(0, np.gradient(level) * -80), 0.05, 1.8)
    seepage = _series(n, rng, base=18.5, amp=2.5, noise=0.6, drift=0.5, floor=8.0, ceil=45.0, phase=1.2)
    # CCTV: mostly online; occasional dropouts; motion bursts at night-ish
    online = np.ones(n)
    dropout = rng.random(n) < 0.012
    online[dropout] = 0.0
    # Cluster dropouts
    for i in np.where(dropout)[0]:
        online[i : min(n, i + int(rng.integers(1, 4)))] = 0.0
    hour = timestamps.hour + timestamps.minute / 60.0
    night = ((hour >= 20) | (hour <= 5)).astype(float)
    motion = rng.poisson(0.15 + 0.55 * night)
    motion = np.where(online < 0.5, 0, motion).astype(float)
    frame_age = np.where(online > 0.5, rng.uniform(2, 25, n), rng.uniform(120, 900, n))

    return {
        "hydro": (
            _rows_for_channel(
                timestamps, level, station_id=station_id, domain="hydro",
                channel="water_level_m", unit="m",
            )
            + _rows_for_channel(
                timestamps, velocity, station_id=station_id, domain="hydro",
                channel="water_velocity_ms", unit="m/s",
            )
        ),
        "seepage": _rows_for_channel(
            timestamps, seepage, station_id=station_id, domain="seepage",
            channel="seepage_lpm", unit="L/min",
        ),
        "cctv": (
            _rows_for_channel(
                timestamps, online, station_id=station_id, domain="cctv",
                channel="camera_online", unit="0/1",
            )
            + _rows_for_channel(
                timestamps, motion, station_id=station_id, domain="cctv",
                channel="motion_events", unit="count/5min",
            )
            + _rows_for_channel(
                timestamps, frame_age, station_id=station_id, domain="cctv",
                channel="last_frame_age_s", unit="s",
            )
        ),
    }


def simulate_issa_soil(
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    station_id: str,
) -> dict[str, list[dict[str, Any]]]:
    n = len(timestamps)
    moisture = _series(n, rng, base=34.0, amp=6.0, noise=0.8, drift=-1.5, floor=12.0, ceil=55.0)
    # Light "irrigation / rain" bumps
    for _ in range(3):
        start = int(rng.integers(0, max(1, n - 36)))
        bump = rng.uniform(6, 14)
        moisture[start : start + 24] += bump * np.linspace(1, 0.2, 24)
    moisture = np.clip(moisture, 12.0, 58.0)
    ec = 0.45 + 0.012 * (moisture - 30) + rng.normal(0, 0.04, n)
    ec = np.clip(ec, 0.15, 1.8)
    ph = _series(n, rng, base=6.3, amp=0.12, noise=0.05, drift=0.02, floor=5.4, ceil=7.4, phase=0.4)
    # NPK: slow-moving nutrients; N more volatile after moisture bumps
    nitrogen = _series(n, rng, base=42.0, amp=3.0, noise=1.2, drift=-0.8, floor=15.0, ceil=90.0)
    nitrogen = nitrogen + 0.15 * np.maximum(0, moisture - 38)
    phosphorus = _series(n, rng, base=28.0, amp=1.5, noise=0.7, drift=-0.2, floor=10.0, ceil=60.0, phase=0.8)
    potassium = _series(n, rng, base=55.0, amp=2.0, noise=0.9, drift=-0.3, floor=20.0, ceil=110.0, phase=1.5)

    rows = []
    for channel, values, unit in (
        ("soil_moisture_pct", moisture, "%"),
        ("soil_ec_ms_cm", ec, "mS/cm"),
        ("soil_ph", ph, "pH"),
        ("soil_n_mg_kg", nitrogen, "mg/kg"),
        ("soil_p_mg_kg", phosphorus, "mg/kg"),
        ("soil_k_mg_kg", potassium, "mg/kg"),
    ):
        rows.extend(
            _rows_for_channel(
                timestamps, values, station_id=station_id, domain="soil",
                channel=channel, unit=unit,
            )
        )
    return {"soil": rows}


def simulate_project_telemetry(
    project: dict[str, Any],
    *,
    cfg: dict | None = None,
    hours: int = 24 * 14,
) -> dict[str, list[dict[str, Any]]]:
    """Simulate non-weather domains for a Heimdall project."""
    cfg = cfg or load_config()
    station_id = str(project["station"]["id"])
    interval = int(cfg.get("sampling", {}).get("interval_minutes", 5))
    seed = int((cfg.get("simulation") or {}).get("seed", 42))
    # Stable but distinct seeds per station
    seed = seed + sum(ord(c) for c in station_id)
    rng = np.random.default_rng(seed)
    timestamps = _timestamps(hours, interval)

    domains = set(project_domains(project))
    out: dict[str, list[dict[str, Any]]] = {}
    kind = str(project.get("kind", "dam"))
    if kind == "dam" or domains & {"hydro", "seepage", "cctv"}:
        hydro_pack = simulate_indra_hydro(timestamps, rng, station_id)
        for domain, rows in hydro_pack.items():
            if domain in domains:
                out[domain] = rows
    if kind == "plantation" or "soil" in domains:
        soil_pack = simulate_issa_soil(timestamps, rng, station_id)
        for domain, rows in soil_pack.items():
            if domain in domains:
                out[domain] = rows

    # Drop weather here — weather continues to use readings/reference tables.
    out.pop("weather", None)
    # Ensure only known channels
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for domain, rows in out.items():
        allowed = set(DOMAIN_CHANNELS.get(domain, {}))
        cleaned[domain] = [r for r in rows if r["channel"] in allowed]
    return cleaned
