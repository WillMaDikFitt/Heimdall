"""Shared hub analysis refresh used by CLI and the Streamlit dashboard."""

from __future__ import annotations

import importlib.util
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import load_config
from src.db import connect
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


def load_refresh_status(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prefer Turso/shared DB status so Streamlit Cloud sees daily Actions runs."""
    try:
        init_db(cfg)
        with connect(cfg) as conn:
            row = conn.execute(
                "SELECT payload_json FROM hub_refresh_status WHERE id = 1"
            ).fetchone()
        if row is not None:
            # Row may be tuple, sqlite3.Row, or mapping depending on driver.
            if isinstance(row, dict):
                raw = row.get("payload_json")
            else:
                try:
                    raw = row["payload_json"]
                except Exception:
                    raw = row[0]
            if raw:
                data = json.loads(str(raw))
                if isinstance(data, dict) and data:
                    return data
    except Exception:
        pass
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_refresh_status(payload: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    """Write status to disk (local/debug) and to the shared DB (Cloud + Actions)."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    STATUS_PATH.write_text(text, encoding="utf-8")
    try:
        init_db(cfg)
        finished = str(payload.get("finished_at") or "")
        ok = 1 if payload.get("ok") else 0
        with connect(cfg) as conn:
            conn.execute(
                """
                INSERT INTO hub_refresh_status (id, payload_json, finished_at, ok)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    finished_at = excluded.finished_at,
                    ok = excluded.ok
                """,
                (text, finished, ok),
            )
    except Exception as exc:
        _log(f"WARN could not persist refresh status to database: {exc}", None)
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
    save_refresh_status(payload, cfg=cfg)
    return payload
