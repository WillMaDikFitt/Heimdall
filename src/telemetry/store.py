"""SQLite persistence for multi-domain Heimdall telemetry."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.ingestion.store import connect, init_db


def ensure_telemetry_table(cfg: dict | None = None) -> None:
    init_db(cfg)
    with connect(cfg) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                station_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                domain TEXT NOT NULL,
                channel TEXT NOT NULL,
                value REAL,
                unit TEXT,
                quality_flag TEXT NOT NULL DEFAULT 'good',
                source TEXT NOT NULL DEFAULT 'simulator',
                meta_json TEXT,
                PRIMARY KEY (station_id, timestamp, domain, channel)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_station_domain_ts "
            "ON telemetry(station_id, domain, timestamp)"
        )


def replace_telemetry_domain(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    station_id: str,
    domain: str,
    cfg: dict | None = None,
) -> int:
    """Replace all rows for one station+domain with a new series."""
    ensure_telemetry_table(cfg)
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)
    sid = str(station_id)
    with connect(cfg) as conn:
        conn.execute(
            "DELETE FROM telemetry WHERE station_id = ? AND domain = ?",
            (sid, str(domain)),
        )
        payload = []
        for row in records:
            meta = row.get("meta") or row.get("meta_json")
            if isinstance(meta, dict):
                meta = json.dumps(meta)
            ts = row["timestamp"]
            if hasattr(ts, "isoformat"):
                ts = pd.Timestamp(ts).tz_convert("UTC").isoformat()
            payload.append(
                {
                    "station_id": sid,
                    "timestamp": str(ts),
                    "domain": str(domain),
                    "channel": str(row["channel"]),
                    "value": None if row.get("value") is None else float(row["value"]),
                    "unit": row.get("unit"),
                    "quality_flag": str(row.get("quality_flag") or "good"),
                    "source": str(row.get("source") or "simulator"),
                    "meta_json": meta,
                }
            )
        if payload:
            conn.executemany(
                """
                INSERT INTO telemetry (
                    station_id, timestamp, domain, channel, value, unit,
                    quality_flag, source, meta_json
                ) VALUES (
                    :station_id, :timestamp, :domain, :channel, :value, :unit,
                    :quality_flag, :source, :meta_json
                )
                """,
                payload,
            )
    return len(payload)


def load_telemetry(
    *,
    station_id: str,
    domain: str | None = None,
    channel: str | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    ensure_telemetry_table(cfg)
    clauses = ["station_id = ?"]
    params: list[Any] = [str(station_id)]
    if domain:
        clauses.append("domain = ?")
        params.append(str(domain))
    if channel:
        clauses.append("channel = ?")
        params.append(str(channel))
    where = " AND ".join(clauses)
    with connect(cfg) as conn:
        frame = pd.read_sql_query(
            f"SELECT * FROM telemetry WHERE {where} ORDER BY timestamp, domain, channel",
            conn,
            params=params,
        )
    if not frame.empty and "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def load_telemetry_latest(
    *,
    station_id: str,
    domain: str | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Latest value per channel for a station (optionally one domain)."""
    frame = load_telemetry(station_id=station_id, domain=domain, cfg=cfg)
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp")
    return frame.groupby(["domain", "channel"], as_index=False).tail(1).reset_index(drop=True)


def load_telemetry_freshness(
    *,
    station_id: str,
    cfg: dict | None = None,
) -> dict[str, pd.Timestamp]:
    """Latest sample time per telemetry domain for a station."""
    ensure_telemetry_table(cfg)
    with connect(cfg) as conn:
        frame = pd.read_sql_query(
            """
            SELECT domain, MAX(timestamp) AS last_timestamp
            FROM telemetry
            WHERE station_id = ?
            GROUP BY domain
            """,
            conn,
            params=[str(station_id)],
        )
    if frame.empty:
        return {}
    out: dict[str, pd.Timestamp] = {}
    for _, row in frame.iterrows():
        ts = pd.to_datetime(row["last_timestamp"], utc=True)
        if pd.isna(ts):
            continue
        out[str(row["domain"])] = ts
    return out
