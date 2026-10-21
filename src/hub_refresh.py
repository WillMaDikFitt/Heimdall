"""Shared hub analysis refresh used by CLI and the Streamlit dashboard."""

from __future__ import annotations

import importlib.util
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import load_config
from src.hub import projects, runtime_config
from src.ingestion.store import (
    init_db,
    replace_forecast_verification_scores,
    replace_imd_daily_rain,
    replace_reference,
)
from src.paths import PROJECT_ROOT
from src.reference.forecast_verify import snapshot_stored_ensemble, verify_archived_forecasts
from src.reference.imd_rain import fetch_imd_daily_rain
from src.reference.open_meteo import fetch_historical

STATUS_PATH = PROJECT_ROOT / "data" / "hub_refresh_status.json"
LogFn = Callable[[str], None]


def _log(message: str, sink: LogFn | None) -> None:
    if sink:
        sink(message)
    else:
        print(message, flush=True)


def load_refresh_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_refresh_status(payload: dict[str, Any]) -> Path:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return STATUS_PATH


def _load_ensemble_script():
    path = PROJECT_ROOT / "scripts" / "run_ensemble_forecast.py"
    spec = importlib.util.spec_from_file_location("run_ensemble_forecast_hub", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_ensemble_forecast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_hub_refresh(
    cfg: dict[str, Any] | None = None,
    *,
    project_ids: list[str] | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """Refresh history, ensemble, archive, and verification for hub projects."""
    cfg = cfg or load_config()
    init_db(cfg)
    all_projects = projects(cfg)
    ids = list(project_ids) if project_ids else list(all_projects.keys())
    steps: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    ok = True
    error: str | None = None
    ensemble_mod = _load_ensemble_script()

    def step(name: str, detail: str, success: bool = True) -> None:
        steps.append({"name": name, "detail": detail, "ok": success})
        _log(("OK " if success else "FAIL ") + f"{name}: {detail}", log)

    try:
        _log(">>> Updating IMD daily rain...", log)
        try:
            imd = fetch_imd_daily_rain(cfg=cfg)
            if imd.empty:
                step("imd", "Not available yet — Open-Meteo verify still runs.")
            else:
                n_imd = replace_imd_daily_rain(imd, cfg=cfg)
                step(
                    "imd",
                    f"Stored {n_imd} daily rows ({imd['date'].min()} -> {imd['date'].max()}).",
                )
        except Exception as exc:
            step("imd", f"Using last stored series ({exc})", success=False)

        for project_id in ids:
            if project_id not in all_projects:
                step(f"project:{project_id}", "Unknown project — skipped", success=False)
                continue
            runtime = runtime_config(cfg, project_id)
            station = runtime["station"]
            station_id = str(station["id"])
            label = all_projects[project_id]["label"]
            _log(f">>> Project {label} ({station_id})", log)

            try:
                reference = fetch_historical(
                    latitude=float(station["latitude"]),
                    longitude=float(station["longitude"]),
                    cfg=runtime,
                    use_cache=False,
                )
                n_ref = replace_reference(reference, cfg=cfg, station_id=station_id)
                step(f"history:{project_id}", f"Stored {n_ref} historical hours.")
            except Exception as exc:
                step(f"history:{project_id}", f"Using last stored series ({exc})", success=False)

            try:
                ensemble_mod.run_ensemble_for_config(runtime, log=lambda m: _log(m, log))
                step(f"ensemble:{project_id}", "Bias-corrected ensemble stored.")
            except Exception as exc:
                ok = False
                error = str(exc)
                step(f"ensemble:{project_id}", str(exc), success=False)
                continue

            try:
                path = snapshot_stored_ensemble(cfg=runtime, station_id=station_id)
                step(f"archive:{project_id}", f"Archived {path.name}")
            except Exception as exc:
                ok = False
                error = str(exc)
                step(f"archive:{project_id}", str(exc), success=False)

        _log(">>> Scoring matured archives...", log)
        try:
            scores = verify_archived_forecasts(cfg=cfg)
            replace_forecast_verification_scores(scores, cfg=cfg)
            step("verify", f"Saved {len(scores)} verification score rows.")
        except Exception as exc:
            ok = False
            error = str(exc)
            step("verify", str(exc), success=False)

    except Exception as exc:
        ok = False
        error = str(exc)
        step("fatal", traceback.format_exc(), success=False)

    finished = datetime.now(timezone.utc).isoformat()
    payload = {
        "started_at": started,
        "finished_at": finished,
        "ok": ok,
        "error": error,
        "project_ids": ids,
        "steps": steps,
    }
    save_refresh_status(payload)
    return payload
