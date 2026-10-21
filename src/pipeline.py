"""Fetch area weather and fit station calibration for a hub project."""

from __future__ import annotations

from typing import Any

from src.calibration.align import align_to_reference, resample_sensor
from src.calibration.fit import fit_all_channels
from src.config import load_config
from src.hub import projects, runtime_config
from src.ingestion.store import (
    init_db,
    insert_coefficients,
    load_readings,
    load_reference,
    replace_calibrated,
    replace_payloads,
    replace_reference,
)
from src.reference.open_meteo import fetch_historical
from src.simulator import simulate_from_reference


def calibrate_project(
    project_id: str,
    cfg: dict[str, Any] | None = None,
    simulate_if_empty: bool = True,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    project = projects(cfg)[project_id]
    runtime = runtime_config(cfg, project_id)
    station = runtime["station"]
    station_id = str(station["id"])
    lat = float(station["latitude"])
    lon = float(station["longitude"])
    area_km = float(station.get("area_radius_km", 12))

    init_db(cfg)
    reference = fetch_historical(latitude=lat, longitude=lon, cfg=runtime)
    n_ref = replace_reference(reference, cfg=cfg, station_id=station_id)

    sensor = load_readings(station_id, cfg=cfg)
    simulated = False
    if sensor.empty and simulate_if_empty:
        payloads = simulate_from_reference(reference, cfg=runtime)
        replace_payloads(payloads, cfg=cfg, station_id=station_id)
        sensor = load_readings(station_id, cfg=cfg)
        simulated = True
    if sensor.empty:
        raise RuntimeError(f"No sensor readings for {station_id}.")

    stored_ref = load_reference(cfg=cfg, station_id=station_id)
    minutes = int(runtime["calibration"]["resample_minutes"])
    hourly = resample_sensor(sensor, minutes=minutes)
    aligned = align_to_reference(hourly, stored_ref)
    if aligned.empty:
        raise RuntimeError(f"Sensor and area weather do not overlap for {station_id}.")

    calibrated, coeffs = fit_all_channels(
        aligned,
        station_id=station_id,
        iqr_multiplier=float(runtime["calibration"]["iqr_multiplier"]),
        holdout_fraction=float(runtime["calibration"].get("holdout_fraction", 0.3)),
    )
    n_cal = replace_calibrated(calibrated, cfg=cfg, station_id=station_id)
    insert_coefficients(coeffs, cfg=cfg)
    return {
        "project_id": project_id,
        "label": project["label"],
        "station_id": station_id,
        "latitude": lat,
        "longitude": lon,
        "area_radius_km": area_km,
        "n_reference": n_ref,
        "n_sensor": len(sensor),
        "n_calibrated": n_cal,
        "simulated": simulated,
        "coeffs": coeffs,
    }


def calibrate_hub(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    return [calibrate_project(project_id, cfg=cfg) for project_id in projects(cfg)]
