"""Domain / channel catalogue for Heimdall projects."""

from __future__ import annotations

from typing import Any

# Chemistry-correct NPK labels (N=nitrogen, P=phosphorus, K=potassium).
DOMAIN_CHANNELS: dict[str, dict[str, dict[str, Any]]] = {
    "weather": {
        "temperature_c": {"label": "Temperature", "unit": "°C"},
        "humidity_pct": {"label": "Humidity", "unit": "%"},
        "pressure_hpa": {"label": "Pressure", "unit": "hPa"},
        "rainfall_mm": {"label": "Rainfall", "unit": "mm"},
        "wind_speed_ms": {"label": "Wind speed", "unit": "m/s"},
        "wind_direction_deg": {"label": "Wind direction", "unit": "deg"},
    },
    "hydro": {
        "water_level_m": {"label": "Water level", "unit": "m"},
        "water_velocity_ms": {"label": "Water velocity", "unit": "m/s"},
    },
    "seepage": {
        "seepage_lpm": {"label": "Seepage flow", "unit": "L/min"},
    },
    "cctv": {
        "camera_online": {"label": "Camera", "unit": ""},
        "motion_events": {"label": "Motion", "unit": "/5 min"},
        "last_frame_age_s": {"label": "Frame age", "unit": "s"},
    },
    "soil": {
        "soil_moisture_pct": {"label": "Moisture", "unit": "%"},
        "soil_ec_ms_cm": {"label": "EC", "unit": "mS/cm"},
        "soil_ph": {"label": "pH", "unit": ""},
        "soil_n_mg_kg": {"label": "N", "unit": "mg/kg"},
        "soil_p_mg_kg": {"label": "P", "unit": "mg/kg"},
        "soil_k_mg_kg": {"label": "K", "unit": "mg/kg"},
    },
}

DOMAIN_LABELS = {
    "weather": "Weather",
    "hydro": "Water",
    "seepage": "Seepage",
    "cctv": "CCTV",
    "soil": "Soil",
}

DEFAULT_PROJECT_DOMAINS = {
    "dam": ["weather", "hydro", "seepage", "cctv"],
    "plantation": ["weather", "soil"],
}


def domain_labels() -> dict[str, str]:
    return dict(DOMAIN_LABELS)


def project_domains(project: dict[str, Any]) -> list[str]:
    raw = project.get("domains")
    if isinstance(raw, list) and raw:
        return [str(d) for d in raw if str(d) in DOMAIN_CHANNELS]
    kind = str(project.get("kind", "dam"))
    return list(DEFAULT_PROJECT_DOMAINS.get(kind, ["weather"]))
