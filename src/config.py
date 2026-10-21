from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.paths import CONFIG_PATH, PROJECT_ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_path"] = str(cfg_path)
    data["open_meteo"]["cache_dir"] = str(_resolve(data["open_meteo"]["cache_dir"]))
    data["storage"]["sqlite_path"] = str(_resolve(data["storage"]["sqlite_path"]))
    return data


def channel_names(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["channels"].keys())


def _write_config(data: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    load_config.cache_clear()


def save_station_location(
    project_id: str,
    latitude: float,
    longitude: float,
    area_radius_km: float | None = None,
) -> None:
    """Write a new map pin and optional forecast-area radius into config.yaml."""
    from src.hub import clamp_area_km

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    lat = round(float(latitude), 5)
    lon = round(float(longitude), 5)
    hub = data.setdefault("hub", {})
    default_id = str(hub.get("default") or "indra")
    projects = hub.setdefault("projects", {})
    # Top-level station holds the default project pin (INDRA).
    if project_id == default_id or project_id in {"indra", "leshka", "myntdu"}:
        target = data.setdefault("station", {})
        kind = str((projects.get(default_id) or {}).get("kind", "dam"))
    else:
        station = projects.setdefault(project_id, {})
        target = station.setdefault("station", {})
        kind = str(station.get("kind", "plantation"))
    target["latitude"] = lat
    target["longitude"] = lon
    if area_radius_km is not None:
        default = 12.0 if kind == "dam" else 6.0
        target["area_radius_km"] = clamp_area_km(area_radius_km, default)
    _write_config(data)



def _sync_streamlit_primary(accent: str) -> None:
    """Keep Streamlit primaryColor aligned with the hub theme accent."""
    toml_path = PROJECT_ROOT / ".streamlit" / "config.toml"
    if not toml_path.exists():
        return
    raw = toml_path.read_text(encoding="utf-8")
    import re

    updated, n = re.subn(
        r'(?m)^primaryColor\s*=\s*".*"$',
        f'primaryColor = "{accent}"',
        raw,
        count=1,
    )
    if n:
        toml_path.write_text(updated, encoding="utf-8")


def save_hub_settings(updates: dict[str, Any]) -> None:
    """Merge hub/forecast/open_meteo settings into config.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    hub = data.setdefault("hub", {})
    if "default" in updates:
        hub["default"] = str(updates["default"])

    ui = hub.setdefault("ui", {})
    for key in (
        "data_window",
        "show_calibrated",
        "show_reference",
        "show_raw",
        "despike",
        "color_theme",
        "auto_refresh_minutes",
    ):
        if key in updates:
            ui[key] = updates[key]

    if "skill" in updates and isinstance(updates["skill"], dict):
        skill = ui.setdefault("skill", {})
        for key, value in updates["skill"].items():
            skill[str(key)] = value

    if "color_theme" in updates:
        try:
            from app.theme import theme_tokens

            _sync_streamlit_primary(str(theme_tokens(updates["color_theme"])["accent"]))
        except Exception:
            pass

    forecast = data.setdefault("forecast", {})
    if "operational_days" in updates:
        forecast["operational_days"] = int(updates["operational_days"])
    if "cache_minutes" in updates:
        forecast["cache_minutes"] = int(updates["cache_minutes"])
    if "bias_window_days" in updates:
        forecast["bias_window_days"] = int(updates["bias_window_days"])
    if "verify_min_age_hours" in updates:
        forecast["verify_min_age_hours"] = int(updates["verify_min_age_hours"])
    if "ensemble_members" in updates:
        members = [str(m) for m in updates["ensemble_members"] if str(m).strip()]
        if not members:
            raise ValueError("Pick at least one forecast model.")
        forecast["ensemble_members"] = members

    calibration = data.setdefault("calibration", {})
    if "resample_minutes" in updates:
        calibration["resample_minutes"] = int(updates["resample_minutes"])
    if "iqr_multiplier" in updates:
        calibration["iqr_multiplier"] = float(updates["iqr_multiplier"])
    if "holdout_fraction" in updates:
        calibration["holdout_fraction"] = float(updates["holdout_fraction"])

    if "drift_thresholds" in updates and isinstance(updates["drift_thresholds"], dict):
        channels = data.setdefault("channels", {})
        for channel, threshold in updates["drift_thresholds"].items():
            if channel not in channels:
                continue
            channels[channel]["drift_alert_threshold"] = float(threshold)

    open_meteo = data.setdefault("open_meteo", {})
    if "historical_days" in updates:
        open_meteo["historical_days"] = int(updates["historical_days"])

    _write_config(data)
