"""Archive issued forecasts and score against later historical truth."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.calibration.qc import metrics
from src.config import load_config
from src.forecast.metrics import daily_rain_metrics, lead_rain_metrics, rain_metrics
from src.ingestion.store import load_ensemble_forecast, load_imd_daily_rain

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CHANNELS = ("temperature_c", "humidity_pct", "rainfall_mm")
IMD_WET_DAY_MM = 2.5


def _archive_dir(cfg: dict) -> Path:
    d = Path(cfg["open_meteo"]["cache_dir"]) / "forecast_archive"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot_forecast(cfg: dict | None = None, forecast_days: int = 3) -> Path:
    """Store today's issued Open-Meteo forecast (secondary baseline)."""
    cfg = cfg or load_config()
    station = cfg["station"]
    params = {
        "latitude": float(station["latitude"]),
        "longitude": float(station["longitude"]),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "UTC",
        "forecast_days": forecast_days,
    }
    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    now = datetime.now(timezone.utc)
    out = _archive_dir(cfg) / f"issued_{now.strftime('%Y%m%dT%H')}Z.json"
    doc = {"issued_at": now.isoformat(), "kind": "open_meteo", "payload": payload}
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def snapshot_stored_ensemble(
    cfg: dict | None = None,
    station_id: str | None = None,
) -> Path:
    """Archive the bias-corrected ensemble currently shown on Overview."""
    cfg = cfg or load_config()
    sid = str(station_id or cfg["station"]["id"])
    ensemble = load_ensemble_forecast(cfg=cfg, station_id=sid)
    if ensemble.empty:
        raise ValueError(f"No ensemble forecast in the database to archive for {sid}")

    now = datetime.now(timezone.utc)
    cols = [c for c in CHANNELS if c in ensemble.columns]
    rows: list[dict] = []
    for _, row in ensemble.sort_values("timestamp").iterrows():
        item = {"timestamp": pd.Timestamp(row["timestamp"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
        for col in cols:
            val = row[col]
            item[col] = None if pd.isna(val) else float(val)
        rows.append(item)

    bias_flag = False
    if "bias_corrected" in ensemble.columns and len(ensemble):
        bias_flag = bool(pd.to_numeric(ensemble["bias_corrected"], errors="coerce").fillna(0).iloc[0])
    source = str(ensemble["source"].iloc[0]) if "source" in ensemble.columns else "ensemble"
    rain_scale = None
    if "rainfall_mm_scale_applied" in ensemble.columns and len(ensemble):
        rain_scale = float(ensemble["rainfall_mm_scale_applied"].iloc[0])

    station = cfg["station"]
    doc = {
        "issued_at": now.isoformat(),
        "kind": "ensemble",
        "station_id": sid,
        "latitude": float(station["latitude"]),
        "longitude": float(station["longitude"]),
        "source": source,
        "bias_corrected": bias_flag,
        "rainfall_mm_scale_applied": rain_scale,
        "hourly": rows,
    }
    safe_sid = sid.replace("/", "_")
    out = _archive_dir(cfg) / f"issued_{now.strftime('%Y%m%dT%H')}Z_{safe_sid}_ensemble.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def _fetch_archive_truth(
    cfg: dict,
    start: str,
    end: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    station = cfg["station"]
    lag = int(cfg.get("open_meteo", {}).get("archive_lag_days", 3))
    latest = (datetime.now(timezone.utc).date() - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
    if end > latest:
        end = latest
    if start > end:
        raise ValueError(f"No archive days available yet ({start} > {end})")
    params = {
        "latitude": float(latitude if latitude is not None else station["latitude"]),
        "longitude": float(longitude if longitude is not None else station["longitude"]),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
    return df.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "precipitation": "rainfall_mm",
        }
    )


def _frame_from_archive_doc(doc: dict, path: Path) -> tuple[pd.DataFrame, str]:
    """Return forecast frame and model label for an archive JSON."""
    kind = str(doc.get("kind") or "")
    if not kind:
        kind = "ensemble" if path.name.endswith("_ensemble.json") else "open_meteo"

    if kind == "ensemble" or path.name.endswith("_ensemble.json"):
        fc = pd.DataFrame(doc.get("hourly") or [])
        if fc.empty:
            return fc, "ensemble_operational"
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        return fc, "ensemble_operational"

    payload = doc.get("payload") or {}
    hourly = payload.get("hourly") or {}
    fc = pd.DataFrame(hourly)
    if fc.empty:
        return fc, "open_meteo_operational"
    fc["timestamp"] = pd.to_datetime(fc.pop("time"), utc=True)
    fc = fc.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "precipitation": "rainfall_mm",
        }
    )
    return fc, "open_meteo_operational"


def _score_channel(
    channel: str,
    timestamps: pd.Series,
    pred: np.ndarray,
    actual: np.ndarray,
    issued_at: datetime,
    model: str,
) -> dict | None:
    ok = np.isfinite(pred) & np.isfinite(actual)
    if ok.sum() < 3:
        return None
    m = metrics(pred[ok], actual[ok])
    row: dict = {
        "issued_at": issued_at.isoformat(),
        "channel": channel,
        "model": model,
        "station_id": None,
        "mae": m["mae"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "n": m["n"],
        "daily_mae": None,
        "csi": None,
        "csi_6h": None,
        "csi_24h": None,
        "hit_rate": None,
        "hit_rate_6h": None,
        "hit_rate_24h": None,
    }
    if channel == "rainfall_mm":
        rain = rain_metrics(pred[ok], actual[ok])
        daily = daily_rain_metrics(timestamps.to_numpy()[ok], pred[ok], actual[ok])
        lead6 = lead_rain_metrics(pred[ok], actual[ok], lead_hours=6)
        lead24 = lead_rain_metrics(pred[ok], actual[ok], lead_hours=24)
        row.update(
            {
                "daily_mae": daily["daily_mae"],
                "csi": rain["csi"],
                "hit_rate": rain["hit_rate"],
                "csi_6h": lead6["csi_6h"],
                "csi_24h": lead24["csi_24h"],
                "hit_rate_6h": lead6["hit_rate_6h"],
                "hit_rate_24h": lead24["hit_rate_24h"],
            }
        )
    return row


def _score_daily_rain_vs_imd(
    fc: pd.DataFrame,
    imd: pd.DataFrame,
    issued_at: datetime,
    model: str = "ensemble_vs_imd",
) -> dict | None:
    """Compare forecast daily rain totals to IMD 0.25° daily rain (rain only)."""
    if fc.empty or imd.empty or "rainfall_mm" not in fc.columns:
        return None

    pred = fc.copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    # IMD days are local calendar days in India; aggregate forecast on IST dates.
    pred["date"] = pred["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    daily_fc = pred.groupby("date", as_index=False)["rainfall_mm"].sum(min_count=1)

    truth = imd.copy()
    truth["date"] = pd.to_datetime(truth["date"]).dt.date
    merged = daily_fc.merge(
        truth[["date", "rainfall_mm"]],
        on="date",
        suffixes=("_fc", "_imd"),
    )
    if merged.empty:
        return None

    x = pd.to_numeric(merged["rainfall_mm_fc"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(merged["rainfall_mm_imd"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None

    m = metrics(x[ok], y[ok])
    # Daily wet/dry skill (IMD has no hourly timing).
    wet_pred = x[ok] >= IMD_WET_DAY_MM
    wet_act = y[ok] >= IMD_WET_DAY_MM
    hits = int(np.sum(wet_pred & wet_act))
    misses = int(np.sum(~wet_pred & wet_act))
    false_alarms = int(np.sum(wet_pred & ~wet_act))
    denom = hits + misses + false_alarms
    csi = float(hits / denom) if denom > 0 else float("nan")
    hit_rate = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")

    return {
        "issued_at": issued_at.isoformat(),
        "channel": "rainfall_mm",
        "model": model,
        "station_id": None,
        "mae": m["mae"],
        "rmse": m["rmse"],
        "bias": m["bias"],
        "n": m["n"],
        "daily_mae": m["mae"],
        "csi": csi,
        "csi_6h": None,
        "csi_24h": None,
        "hit_rate": hit_rate,
        "hit_rate_6h": None,
        "hit_rate_24h": None,
    }


def verify_archived_forecasts(
    cfg: dict | None = None,
    min_age_hours: int | None = None,
) -> list[dict]:
    """
    Score archived issued forecasts where enough time has passed for archive truth.

    - Hourly channels vs Open-Meteo archive (temperature, humidity, rain)
    - Daily rain totals vs IMD 0.25° grid when IMD rows are available
    """
    cfg = cfg or load_config()
    if min_age_hours is None:
        min_age_hours = int(cfg.get("forecast", {}).get("verify_min_age_hours", 48))
    archive = _archive_dir(cfg)
    scores: list[dict] = []
    now = datetime.now(timezone.utc)
    imd = load_imd_daily_rain(cfg=cfg)

    for path in sorted(archive.glob("issued_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        issued_at = datetime.fromisoformat(doc["issued_at"])
        if issued_at.tzinfo is None:
            issued_at = issued_at.replace(tzinfo=timezone.utc)
        if (now - issued_at).total_seconds() < min_age_hours * 3600:
            continue

        fc, model = _frame_from_archive_doc(doc, path)
        if fc.empty or "timestamp" not in fc.columns:
            continue

        station_id = str(doc.get("station_id") or cfg["station"]["id"])
        lat = doc.get("latitude")
        lon = doc.get("longitude")
        start = fc["timestamp"].min().strftime("%Y-%m-%d")
        end = fc["timestamp"].max().strftime("%Y-%m-%d")
        try:
            truth = _fetch_archive_truth(
                cfg,
                start,
                end,
                latitude=float(lat) if lat is not None else None,
                longitude=float(lon) if lon is not None else None,
            )
        except Exception:
            truth = pd.DataFrame()

        if not truth.empty:
            merged = fc.merge(truth, on="timestamp", suffixes=("_fc", "_truth"))
            for col in CHANNELS:
                fc_col = f"{col}_fc" if f"{col}_fc" in merged.columns else col
                truth_col = f"{col}_truth" if f"{col}_truth" in merged.columns else None
                if truth_col is None or fc_col not in merged.columns:
                    continue
                pred = pd.to_numeric(merged[fc_col], errors="coerce").to_numpy(dtype=float)
                actual = pd.to_numeric(merged[truth_col], errors="coerce").to_numpy(dtype=float)
                scored = _score_channel(col, merged["timestamp"], pred, actual, issued_at, model)
                if scored is not None:
                    scored["station_id"] = station_id
                    scores.append(scored)

        # Second referee: IMD daily rain (ensemble / open-meteo outlooks).
        if not imd.empty and "rainfall_mm" in fc.columns:
            imd_model = (
                "ensemble_vs_imd"
                if model == "ensemble_operational"
                else "open_meteo_vs_imd"
            )
            imd_score = _score_daily_rain_vs_imd(fc, imd, issued_at, model=imd_model)
            if imd_score is not None:
                imd_score["station_id"] = station_id
                scores.append(imd_score)

    return scores
