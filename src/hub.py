"""Heimdall projects, map pins, and forecast-area settings."""

from __future__ import annotations

from typing import Any

KIND_AREA_KM = {
    "dam": 12.0,
    "plantation": 6.0,
}


def clamp_area_km(value: object, default: float = 12.0) -> float:
    try:
        area = float(value)
    except (TypeError, ValueError):
        area = default
    return round(min(50.0, max(1.0, area)), 1)


def _with_area(station: dict[str, Any], kind: str) -> dict[str, Any]:
    out = dict(station)
    default = KIND_AREA_KM.get(kind, 12.0)
    out["area_radius_km"] = clamp_area_km(out.get("area_radius_km"), default)
    return out


def product_name(cfg: dict[str, Any]) -> str:
    product = cfg.get("product") or {}
    return str(product.get("name") or "Heimdall")


def product_tagline(cfg: dict[str, Any]) -> str:
    product = cfg.get("product") or {}
    return str(product.get("tagline") or "Sensor data hub")


def projects(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build project map from hub.projects YAML (INDRA / ISSA, etc.)."""
    base_station = dict(cfg["station"])
    hub = cfg.get("hub") or {}
    spec = dict(hub.get("projects") or {})
    default_id = str(hub.get("default") or next(iter(spec), "indra"))

    out: dict[str, dict[str, Any]] = {}
    for project_id, raw in spec.items():
        raw = dict(raw or {})
        kind = str(raw.get("kind", "dam"))
        nested = raw.get("station")
        if nested:
            station = _with_area(dict(nested), kind)
        elif project_id == default_id or not out:
            station = _with_area(base_station, kind)
        else:
            station = _with_area(dict(base_station), kind)
        domains = raw.get("domains")
        out[str(project_id)] = {
            "id": str(project_id),
            "kind": kind,
            "label": str(raw.get("label", project_id.upper())),
            "blurb": str(raw.get("blurb", "")),
            "station": station,
            "domains": list(domains) if isinstance(domains, list) else None,
        }

    if not out:
        # Absolute fallback
        out["indra"] = {
            "id": "indra",
            "kind": "dam",
            "label": "INDRA",
            "blurb": "Hydro & weather monitoring",
            "station": _with_area(base_station, "dam"),
            "domains": ["weather", "hydro", "seepage", "cctv"],
        }
    return out


def resolve_project_id(cfg: dict[str, Any], project_id: str | None) -> str:
    all_projects = projects(cfg)
    hub = cfg.get("hub") or {}
    default = str(hub.get("default") or "indra")
    aliases = {
        "leshka": "indra",
        "cardamom": "issa",
        "myntdu": "indra",
    }
    if project_id:
        pid = aliases.get(project_id, project_id)
        if pid in all_projects:
            return pid
    if default in all_projects:
        return default
    return next(iter(all_projects))


def runtime_config(cfg: dict[str, Any], project_id: str) -> dict[str, Any]:
    pid = resolve_project_id(cfg, project_id)
    project = projects(cfg)[pid]
    return {**cfg, "station": project["station"]}
