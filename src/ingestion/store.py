"""SQLite / Turso persistence for raw sensor, reference, calibrated, and coefficient rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import load_config
from src.db import connect, db_path
from src.schema import CHANNEL_NAMES, SensorPayload

READINGS_COLUMNS = [
    "station_id",
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "rainfall_mm",
    "wind_speed_ms",
    "wind_direction_deg",
    "firmware",
    "battery_v",
    "source",
]

__all__ = [
    "READINGS_COLUMNS",
    "CHANNEL_NAMES",
    "SensorPayload",
    "connect",
    "db_path",
    "init_db",
]


_SCHEMA_READY = False


def init_db(cfg: dict | None = None, *, force: bool = False) -> None:
    """Create/migrate schema. Skips repeat work in the same process (big Turso win)."""
    global _SCHEMA_READY
    if _SCHEMA_READY and not force:
        return
    with connect(cfg) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                rainfall_mm REAL,
                wind_speed_ms REAL,
                wind_direction_deg REAL,
                firmware TEXT,
                battery_v REAL,
                source TEXT NOT NULL DEFAULT 'sensor'
            );
            CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(station_id, timestamp);

            CREATE TABLE IF NOT EXISTS reference_readings (
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                rainfall_mm REAL,
                wind_speed_ms REAL,
                wind_direction_deg REAL,
                PRIMARY KEY (timestamp, station_id)
            );

            CREATE TABLE IF NOT EXISTS calibrated_readings (
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                raw REAL,
                calibrated REAL,
                reference_value REAL,
                quality_flag TEXT NOT NULL,
                PRIMARY KEY (timestamp, station_id, channel)
            );

            CREATE TABLE IF NOT EXISTS calibration_coefficients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                scale REAL NOT NULL,
                offset REAL NOT NULL,
                fitted_at TEXT NOT NULL,
                n_points INTEGER NOT NULL,
                mae REAL,
                rmse REAL,
                bias REAL,
                mae_raw REAL,
                rmse_raw REAL,
                bias_raw REAL
            );

            CREATE TABLE IF NOT EXISTS forecast_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                model TEXT NOT NULL,
                train_start TEXT,
                train_end TEXT,
                forecast_start TEXT,
                forecast_end TEXT,
                train_hours INTEGER,
                forecast_hours INTEGER,
                mae REAL,
                rmse REAL,
                bias REAL,
                n INTEGER,
                fitted_at TEXT,
                train_mode TEXT
            );

            CREATE TABLE IF NOT EXISTS forecast_predictions (
                timestamp TEXT NOT NULL,
                channel TEXT NOT NULL,
                model TEXT NOT NULL,
                predicted REAL,
                actual REAL,
                PRIMARY KEY (timestamp, channel, model)
            );

            CREATE TABLE IF NOT EXISTS walkforward_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                model TEXT NOT NULL,
                lead_hours INTEGER NOT NULL,
                mae REAL,
                mae_std REAL,
                rmse REAL,
                bias REAL,
                n INTEGER,
                skill_vs_naive REAL,
                fitted_at TEXT,
                train_mode TEXT
            );

            CREATE TABLE IF NOT EXISTS power_readings (
                timestamp TEXT PRIMARY KEY,
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                rainfall_mm REAL,
                wind_speed_ms REAL,
                wind_direction_deg REAL
            );

            CREATE TABLE IF NOT EXISTS imd_daily_rain (
                date TEXT PRIMARY KEY,
                rainfall_mm REAL,
                grid_lat REAL,
                grid_lon REAL
            );

            CREATE TABLE IF NOT EXISTS source_validation_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_a TEXT NOT NULL,
                source_b TEXT NOT NULL,
                channel TEXT NOT NULL,
                granularity TEXT NOT NULL,
                mae REAL,
                rmse REAL,
                bias REAL,
                n INTEGER,
                computed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS forecast_verification_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issued_at TEXT,
                channel TEXT NOT NULL,
                model TEXT NOT NULL,
                mae REAL,
                rmse REAL,
                bias REAL,
                n INTEGER,
                computed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS ensemble_forecast_hourly (
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL DEFAULT 'WS-INDRA',
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                rainfall_mm REAL,
                wind_speed_ms REAL,
                wind_direction_deg REAL,
                temperature_c_spread REAL,
                humidity_pct_spread REAL,
                pressure_hpa_spread REAL,
                rainfall_mm_spread REAL,
                source TEXT,
                fetched_at TEXT,
                bias_corrected INTEGER DEFAULT 0,
                PRIMARY KEY (timestamp, station_id)
            );

            CREATE TABLE IF NOT EXISTS nwp_member_hourly (
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL DEFAULT 'WS-INDRA',
                member TEXT NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                rainfall_mm REAL,
                wind_speed_ms REAL,
                wind_direction_deg REAL,
                fetched_at TEXT,
                PRIMARY KEY (timestamp, station_id, member)
            );
            """
        )
        _migrate_schema(conn)
    _SCHEMA_READY = True


def _migrate_schema(conn: Any) -> None:
    """Add columns/tables introduced after initial schema without breaking old DBs."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(calibration_coefficients)")}
    for col, typedef in [
        ("mae_holdout", "REAL"),
        ("rmse_holdout", "REAL"),
        ("bias_holdout", "REAL"),
        ("n_holdout", "INTEGER"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE calibration_coefficients ADD COLUMN {col} {typedef}")

    score_cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_scores)")}
    for col, typedef in [
        ("csi", "REAL"),
        ("wet_hour_bias", "REAL"),
        ("hit_rate", "REAL"),
        ("daily_mae", "REAL"),
        ("block_24h_mae", "REAL"),
        ("block_6h_mae", "REAL"),
        ("csi_6h", "REAL"),
        ("csi_24h", "REAL"),
        ("hit_rate_6h", "REAL"),
        ("hit_rate_24h", "REAL"),
    ]:
        if col not in score_cols:
            conn.execute(f"ALTER TABLE forecast_scores ADD COLUMN {col} {typedef}")

    verify_cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_verification_scores)")}
    for col, typedef in [
        ("daily_mae", "REAL"),
        ("csi", "REAL"),
        ("csi_6h", "REAL"),
        ("csi_24h", "REAL"),
        ("hit_rate", "REAL"),
        ("hit_rate_6h", "REAL"),
        ("hit_rate_24h", "REAL"),
    ]:
        if col not in verify_cols:
            conn.execute(f"ALTER TABLE forecast_verification_scores ADD COLUMN {col} {typedef}")
    verify_cols = {row[1] for row in conn.execute("PRAGMA table_info(forecast_verification_scores)")}
    if "station_id" not in verify_cols:
        conn.execute(
            "ALTER TABLE forecast_verification_scores ADD COLUMN station_id TEXT DEFAULT 'WS-INDRA'"
        )
    _migrate_reference_station_id(conn)
    _migrate_ensemble_station_id(conn)
    _migrate_nwp_station_id(conn)
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hub_refresh_status (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            payload_json TEXT NOT NULL,
            finished_at TEXT,
            ok INTEGER
        )
        """
    )


def _migrate_ensemble_station_id(conn: Any) -> None:
    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "ensemble_forecast_hourly" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(ensemble_forecast_hourly)")]
    if "station_id" in columns:
        return
    conn.execute(
        """
        CREATE TABLE ensemble_forecast_hourly_v2 (
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            temperature_c REAL,
            humidity_pct REAL,
            pressure_hpa REAL,
            rainfall_mm REAL,
            wind_speed_ms REAL,
            wind_direction_deg REAL,
            temperature_c_spread REAL,
            humidity_pct_spread REAL,
            pressure_hpa_spread REAL,
            rainfall_mm_spread REAL,
            source TEXT,
            fetched_at TEXT,
            bias_corrected INTEGER DEFAULT 0,
            PRIMARY KEY (timestamp, station_id)
        )
        """
    )
    rest = [c for c in columns if c != "timestamp"]
    rest_sql = ", ".join(rest)
    conn.execute(
        f"""
        INSERT INTO ensemble_forecast_hourly_v2 (timestamp, station_id, {rest_sql})
        SELECT timestamp, 'WS-INDRA', {rest_sql} FROM ensemble_forecast_hourly
        """
    )
    conn.execute("DROP TABLE ensemble_forecast_hourly")
    conn.execute("ALTER TABLE ensemble_forecast_hourly_v2 RENAME TO ensemble_forecast_hourly")


def _migrate_nwp_station_id(conn: Any) -> None:
    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "nwp_member_hourly" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(nwp_member_hourly)")]
    if "station_id" in columns:
        return
    conn.execute(
        """
        CREATE TABLE nwp_member_hourly_v2 (
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            member TEXT NOT NULL,
            temperature_c REAL,
            humidity_pct REAL,
            pressure_hpa REAL,
            rainfall_mm REAL,
            wind_speed_ms REAL,
            wind_direction_deg REAL,
            fetched_at TEXT,
            PRIMARY KEY (timestamp, station_id, member)
        )
        """
    )
    rest = [c for c in columns if c != "timestamp"]
    rest_sql = ", ".join(rest)
    conn.execute(
        f"""
        INSERT INTO nwp_member_hourly_v2 (timestamp, station_id, {rest_sql})
        SELECT timestamp, 'WS-INDRA', {rest_sql} FROM nwp_member_hourly
        """
    )
    conn.execute("DROP TABLE nwp_member_hourly")
    conn.execute("ALTER TABLE nwp_member_hourly_v2 RENAME TO nwp_member_hourly")


def _migrate_reference_station_id(conn: Any) -> None:
    """Older databases stored one Open-Meteo series; tag it as the dam station."""
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "reference_readings" not in tables:
        return
    columns = [row[1] for row in conn.execute("PRAGMA table_info(reference_readings)")]
    if "station_id" in columns:
        return
    channel_sql = ", ".join(CHANNEL_NAMES)
    conn.execute(
        f"""
        CREATE TABLE reference_readings_v2 (
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            {", ".join(f"{name} REAL" for name in CHANNEL_NAMES)},
            PRIMARY KEY (timestamp, station_id)
        )
        """
    )
    conn.execute(
        f"""
        INSERT INTO reference_readings_v2 (timestamp, station_id, {channel_sql})
        SELECT timestamp, 'WS-INDRA', {channel_sql} FROM reference_readings
        """
    )
    conn.execute("DROP TABLE reference_readings")
    conn.execute("ALTER TABLE reference_readings_v2 RENAME TO reference_readings")


def insert_payload(payload: SensorPayload, cfg: dict | None = None) -> int:
    init_db(cfg)
    row = payload.to_row()
    with connect(cfg) as conn:
        cur = conn.execute(
            f"""
            INSERT INTO readings ({", ".join(READINGS_COLUMNS)})
            VALUES ({", ".join(":" + c for c in READINGS_COLUMNS)})
            """,
            row,
        )
        return int(cur.lastrowid)


def insert_payloads(payloads: list[SensorPayload], cfg: dict | None = None) -> int:
    init_db(cfg)
    rows = [p.to_row() for p in payloads]
    if not rows:
        return 0
    with connect(cfg) as conn:
        conn.executemany(
            f"""
            INSERT INTO readings ({", ".join(READINGS_COLUMNS)})
            VALUES ({", ".join(":" + c for c in READINGS_COLUMNS)})
            """,
            rows,
        )
        return len(rows)


def replace_payloads(
    payloads: list[SensorPayload],
    cfg: dict | None = None,
    station_id: str | None = None,
) -> int:
    """Replace simulated readings for one station, leaving the other project intact."""
    init_db(cfg)
    rows = [p.to_row() for p in payloads]
    if station_id is None:
        if not rows:
            raise ValueError("station_id is required when replacing empty readings")
        station_id = str(rows[0]["station_id"])
    with connect(cfg) as conn:
        conn.execute("DELETE FROM readings WHERE station_id = ?", (station_id,))
        if not rows:
            return 0
        conn.executemany(
            f"""
            INSERT INTO readings ({", ".join(READINGS_COLUMNS)})
            VALUES ({", ".join(":" + c for c in READINGS_COLUMNS)})
            """,
            rows,
        )
        return len(rows)


def replace_reference(
    frame: pd.DataFrame,
    cfg: dict | None = None,
    station_id: str | None = None,
) -> int:
    init_db(cfg)
    df = frame.copy()
    if station_id is None:
        if "station_id" in df.columns and not df.empty:
            station_id = str(df["station_id"].iloc[0])
        else:
            raise ValueError("station_id is required when storing reference weather")
    df["station_id"] = station_id
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = ["timestamp", "station_id", *CHANNEL_NAMES]
    records = df[cols].to_dict(orient="records")
    with connect(cfg) as conn:
        conn.execute("DELETE FROM reference_readings WHERE station_id = ?", (station_id,))
        conn.executemany(
            f"""
            INSERT INTO reference_readings ({", ".join(cols)})
            VALUES ({", ".join(":" + c for c in cols)})
            """,
            records,
        )
        return len(records)


def replace_calibrated(
    frame: pd.DataFrame,
    cfg: dict | None = None,
    station_id: str | None = None,
) -> int:
    init_db(cfg)
    df = frame.copy()
    if station_id is None:
        if "station_id" in df.columns and not df.empty:
            station_id = str(df["station_id"].iloc[0])
        else:
            raise ValueError("station_id is required when storing calibrated readings")
    df["station_id"] = station_id
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = [
        "timestamp",
        "station_id",
        "channel",
        "raw",
        "calibrated",
        "reference_value",
        "quality_flag",
    ]
    records = df[cols].to_dict(orient="records")
    with connect(cfg) as conn:
        conn.execute("DELETE FROM calibrated_readings WHERE station_id = ?", (station_id,))
        conn.executemany(
            f"""
            INSERT INTO calibrated_readings ({", ".join(cols)})
            VALUES ({", ".join(":" + c for c in cols)})
            """,
            records,
        )
        return len(records)


def insert_coefficients(rows: list[dict], cfg: dict | None = None) -> None:
    init_db(cfg)
    prepared: list[dict] = []
    for row in rows:
        item = dict(row)
        fitted = item["fitted_at"]
        if isinstance(fitted, datetime):
            if fitted.tzinfo is None:
                fitted = fitted.replace(tzinfo=timezone.utc)
            item["fitted_at"] = fitted.isoformat()
        for key in ("mae_holdout", "rmse_holdout", "bias_holdout", "n_holdout"):
            item.setdefault(key, None)
        prepared.append(item)
    with connect(cfg) as conn:
        conn.executemany(
            """
            INSERT INTO calibration_coefficients (
                station_id, channel, scale, offset, fitted_at, n_points,
                mae, rmse, bias, mae_raw, rmse_raw, bias_raw,
                mae_holdout, rmse_holdout, bias_holdout, n_holdout
            ) VALUES (
                :station_id, :channel, :scale, :offset, :fitted_at, :n_points,
                :mae, :rmse, :bias, :mae_raw, :rmse_raw, :bias_raw,
                :mae_holdout, :rmse_holdout, :bias_holdout, :n_holdout
            )
            """,
            prepared,
        )


def load_hub_snapshot(
    station_id: str,
    cfg: dict | None = None,
    *,
    days: int = 14,
) -> dict[str, Any]:
    """One Turso/SQLite session for the dashboard home path (far fewer round-trips)."""
    init_db(cfg)
    cfg = cfg or load_config()
    sid = str(station_id)
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    empty = pd.DataFrame()
    with connect(cfg) as conn:
        ensemble = pd.read_sql_query(
            "SELECT * FROM ensemble_forecast_hourly WHERE station_id = ? ORDER BY timestamp",
            conn,
            params=(sid,),
        )
        readings = pd.read_sql_query(
            "SELECT * FROM readings WHERE station_id = ? AND timestamp >= ? ORDER BY timestamp",
            conn,
            params=(sid, since),
        )
        # Practice data can sit idle for weeks — still show the latest samples.
        if readings.empty:
            readings = pd.read_sql_query(
                "SELECT * FROM readings WHERE station_id = ? ORDER BY timestamp DESC LIMIT 500",
                conn,
                params=(sid,),
            )
            if not readings.empty:
                readings = readings.iloc[::-1].reset_index(drop=True)
        reference = pd.read_sql_query(
            "SELECT * FROM reference_readings WHERE station_id = ? AND timestamp >= ? "
            "ORDER BY timestamp",
            conn,
            params=(sid, since),
        )
        coeffs = pd.read_sql_query(
            """
            SELECT c.*
            FROM calibration_coefficients c
            JOIN (
                SELECT channel, MAX(id) AS max_id
                FROM calibration_coefficients
                WHERE station_id = ?
                GROUP BY channel
            ) latest ON c.id = latest.max_id
            ORDER BY c.channel
            """,
            conn,
            params=(sid,),
        )
        status_row = conn.execute(
            "SELECT payload_json FROM hub_refresh_status WHERE id = 1"
        ).fetchone()
        tel_fresh = pd.read_sql_query(
            """
            SELECT domain, MAX(timestamp) AS last_timestamp
            FROM telemetry
            WHERE station_id = ?
            GROUP BY domain
            """,
            conn,
            params=(sid,),
        )

    if not ensemble.empty and "timestamp" in ensemble.columns:
        ensemble["timestamp"] = pd.to_datetime(ensemble["timestamp"], utc=True)
        if "fetched_at" in ensemble.columns:
            ensemble["fetched_at"] = pd.to_datetime(ensemble["fetched_at"], utc=True)
    if not readings.empty and "timestamp" in readings.columns:
        readings["timestamp"] = pd.to_datetime(readings["timestamp"], utc=True)
    if not reference.empty and "timestamp" in reference.columns:
        reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True)

    refresh_status: dict[str, Any] = {}
    if status_row is not None:
        try:
            import json

            raw = status_row[0]
            if isinstance(status_row, dict):
                raw = status_row.get("payload_json")
            if raw:
                parsed = json.loads(str(raw))
                if isinstance(parsed, dict):
                    refresh_status = parsed
        except Exception:
            refresh_status = {}

    tel_map: dict[str, pd.Timestamp] = {}
    if not tel_fresh.empty:
        for _, row in tel_fresh.iterrows():
            ts = pd.to_datetime(row["last_timestamp"], utc=True)
            if pd.notna(ts):
                tel_map[str(row["domain"])] = ts

    return {
        "ensemble": ensemble if not ensemble.empty else empty,
        "readings": readings if not readings.empty else empty,
        "reference": reference if not reference.empty else empty,
        "coeffs": coeffs if not coeffs.empty else empty,
        "refresh_status": refresh_status,
        "telemetry_freshness": tel_map,
    }


def load_readings(
    station_id: str | None = None,
    cfg: dict | None = None,
    *,
    days: int | None = None,
) -> pd.DataFrame:
    init_db(cfg)
    clauses: list[str] = []
    params: list[Any] = []
    if station_id:
        clauses.append("station_id = ?")
        params.append(str(station_id))
    if days is not None and days > 0:
        since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        clauses.append("timestamp >= ?")
        params.append(since)
    query = "SELECT * FROM readings"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp"
    with connect(cfg) as conn:
        df = pd.read_sql_query(query, conn, params=tuple(params))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_reference(
    cfg: dict | None = None,
    station_id: str | None = None,
    *,
    days: int | None = None,
) -> pd.DataFrame:
    init_db(cfg)
    clauses: list[str] = []
    params: list[Any] = []
    if station_id:
        clauses.append("station_id = ?")
        params.append(str(station_id))
    if days is not None and days > 0:
        since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        clauses.append("timestamp >= ?")
        params.append(since)
    query = "SELECT * FROM reference_readings"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp"
    with connect(cfg) as conn:
        df = pd.read_sql_query(query, conn, params=tuple(params))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_calibrated(cfg: dict | None = None, station_id: str | None = None) -> pd.DataFrame:
    init_db(cfg)
    query = "SELECT * FROM calibrated_readings"
    params: tuple = ()
    if station_id:
        query += " WHERE station_id = ?"
        params = (station_id,)
    query += " ORDER BY timestamp"
    with connect(cfg) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_latest_coefficients(cfg: dict | None = None, station_id: str | None = None) -> pd.DataFrame:
    init_db(cfg)
    where = "WHERE station_id = ?" if station_id else ""
    params: tuple = (station_id,) if station_id else ()
    with connect(cfg) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT c.*
            FROM calibration_coefficients c
            JOIN (
                SELECT channel, MAX(id) AS max_id
                FROM calibration_coefficients
                {where}
                GROUP BY channel
            ) latest ON c.id = latest.max_id
            ORDER BY c.channel
            """,
            conn,
            params=params,
        )
    return df


def replace_forecast_results(
    predictions: pd.DataFrame,
    scores: list[dict],
    cfg: dict | None = None,
) -> None:
    init_db(cfg)
    pred = predictions.copy()
    if not pred.empty:
        pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    with connect(cfg) as conn:
        conn.execute("DELETE FROM forecast_predictions")
        conn.execute("DELETE FROM forecast_scores")
        if scores:
            for row in scores:
                row.setdefault("csi", None)
                row.setdefault("wet_hour_bias", None)
                row.setdefault("hit_rate", None)
                row.setdefault("daily_mae", None)
                row.setdefault("block_24h_mae", None)
                row.setdefault("block_6h_mae", None)
                row.setdefault("csi_6h", None)
                row.setdefault("csi_24h", None)
                row.setdefault("hit_rate_6h", None)
                row.setdefault("hit_rate_24h", None)
            conn.executemany(
                """
                INSERT INTO forecast_scores (
                    channel, model, train_start, train_end, forecast_start, forecast_end,
                    train_hours, forecast_hours, mae, rmse, bias, n, fitted_at, train_mode,
                    csi, wet_hour_bias, hit_rate, daily_mae, block_24h_mae, block_6h_mae,
                    csi_6h, csi_24h, hit_rate_6h, hit_rate_24h
                ) VALUES (
                    :channel, :model, :train_start, :train_end, :forecast_start, :forecast_end,
                    :train_hours, :forecast_hours, :mae, :rmse, :bias, :n, :fitted_at, :train_mode,
                    :csi, :wet_hour_bias, :hit_rate, :daily_mae, :block_24h_mae, :block_6h_mae,
                    :csi_6h, :csi_24h, :hit_rate_6h, :hit_rate_24h
                )
                """,
                scores,
            )
        if not pred.empty:
            cols = ["timestamp", "channel", "model", "predicted", "actual"]
            conn.executemany(
                f"""
                INSERT INTO forecast_predictions ({", ".join(cols)})
                VALUES ({", ".join(":" + c for c in cols)})
                """,
                pred[cols].to_dict(orient="records"),
            )


def load_forecast_scores(cfg: dict | None = None) -> pd.DataFrame:
    init_db(cfg)
    with connect(cfg) as conn:
        return pd.read_sql_query(
            "SELECT * FROM forecast_scores ORDER BY channel, mae",
            conn,
        )


def load_forecast_predictions(cfg: dict | None = None) -> pd.DataFrame:
    init_db(cfg)
    with connect(cfg) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM forecast_predictions ORDER BY timestamp",
            conn,
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def replace_power_readings(frame: pd.DataFrame, cfg: dict | None = None) -> int:
    init_db(cfg)
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = ["timestamp", *CHANNEL_NAMES]
    records = df[cols].to_dict(orient="records")
    with connect(cfg) as conn:
        conn.execute("DELETE FROM power_readings")
        conn.executemany(
            f"INSERT INTO power_readings ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
            records,
        )
        return len(records)


def load_power_readings(cfg: dict | None = None) -> pd.DataFrame:
    init_db(cfg)
    with connect(cfg) as conn:
        df = pd.read_sql_query("SELECT * FROM power_readings ORDER BY timestamp", conn)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def replace_imd_daily_rain(frame: pd.DataFrame, cfg: dict | None = None) -> int:
    init_db(cfg)
    if frame.empty:
        return 0
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["date", "rainfall_mm", "grid_lat", "grid_lon"]
    records = df[cols].to_dict(orient="records")
    with connect(cfg) as conn:
        conn.execute("DELETE FROM imd_daily_rain")
        conn.executemany(
            f"INSERT INTO imd_daily_rain ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
            records,
        )
        return len(records)


def load_imd_daily_rain(cfg: dict | None = None) -> pd.DataFrame:
    init_db(cfg)
    with connect(cfg) as conn:
        df = pd.read_sql_query("SELECT * FROM imd_daily_rain ORDER BY date", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def replace_source_validation_scores(scores: list[dict], cfg: dict | None = None) -> None:
    init_db(cfg)
    now = datetime.now(timezone.utc).isoformat()
    for row in scores:
        row.setdefault("computed_at", now)
    with connect(cfg) as conn:
        conn.execute("DELETE FROM source_validation_scores")
        if scores:
            conn.executemany(
                """
                INSERT INTO source_validation_scores (
                    source_a, source_b, channel, granularity, mae, rmse, bias, n, computed_at
                ) VALUES (
                    :source_a, :source_b, :channel, :granularity, :mae, :rmse, :bias, :n, :computed_at
                )
                """,
                scores,
            )


def load_source_validation_scores(cfg: dict | None = None) -> pd.DataFrame:
    init_db(cfg)
    with connect(cfg) as conn:
        return pd.read_sql_query(
            "SELECT * FROM source_validation_scores ORDER BY channel, mae", conn
        )


def replace_walkforward_scores(scores: list[dict], cfg: dict | None = None) -> None:
    init_db(cfg)
    with connect(cfg) as conn:
        conn.execute("DELETE FROM walkforward_scores")
        if scores:
            conn.executemany(
                """
                INSERT INTO walkforward_scores (
                    channel, model, lead_hours, mae, mae_std, rmse, bias, n,
                    skill_vs_naive, fitted_at, train_mode
                ) VALUES (
                    :channel, :model, :lead_hours, :mae, :mae_std, :rmse, :bias, :n,
                    :skill_vs_naive, :fitted_at, :train_mode
                )
                """,
                scores,
            )


def load_walkforward_scores(cfg: dict | None = None) -> pd.DataFrame:
    init_db(cfg)
    with connect(cfg) as conn:
        return pd.read_sql_query(
            "SELECT * FROM walkforward_scores ORDER BY channel, lead_hours, mae", conn
        )


def replace_forecast_verification_scores(scores: list[dict], cfg: dict | None = None) -> None:
    init_db(cfg)
    cfg = cfg or load_config()
    default_sid = str(cfg["station"]["id"])
    now = datetime.now(timezone.utc).isoformat()
    for row in scores:
        row.setdefault("computed_at", now)
        row.setdefault("station_id", default_sid)
        row.setdefault("daily_mae", None)
        row.setdefault("csi", None)
        row.setdefault("csi_6h", None)
        row.setdefault("csi_24h", None)
        row.setdefault("hit_rate", None)
        row.setdefault("hit_rate_6h", None)
        row.setdefault("hit_rate_24h", None)
    with connect(cfg) as conn:
        conn.execute("DELETE FROM forecast_verification_scores")
        if scores:
            conn.executemany(
                """
                INSERT INTO forecast_verification_scores (
                    issued_at, channel, model, mae, rmse, bias, n, computed_at, station_id,
                    daily_mae, csi, csi_6h, csi_24h, hit_rate, hit_rate_6h, hit_rate_24h
                ) VALUES (
                    :issued_at, :channel, :model, :mae, :rmse, :bias, :n, :computed_at, :station_id,
                    :daily_mae, :csi, :csi_6h, :csi_24h, :hit_rate, :hit_rate_6h, :hit_rate_24h
                )
                """,
                scores,
            )


# Renamed project station IDs (keep Progress history after rebrand).
_STATION_ALIASES = {
    "WS-INDRA": ("WS-INDRA", "WS-LESHKA"),
    "WS-ISSA": ("WS-ISSA", "WS-CARDAMOM"),
}


def _station_id_aliases(station_id: str) -> tuple[str, ...]:
    sid = str(station_id)
    return _STATION_ALIASES.get(sid, (sid,))


def remap_legacy_station_ids(cfg: dict | None = None) -> dict[str, int]:
    """Rewrite LESHKA/CARDAMOM rows to INDRA/ISSA across hub tables."""
    init_db(cfg)
    mapping = (("WS-LESHKA", "WS-INDRA"), ("WS-CARDAMOM", "WS-ISSA"))
    tables = (
        "forecast_verification_scores",
        "ensemble_forecast_hourly",
        "nwp_member_hourly",
        "reference_readings",
        "readings",
        "calibrated_readings",
        "calibration_coefficients",
        "telemetry",
    )
    updated: dict[str, int] = {}
    with connect(cfg) as conn:
        for old, new in mapping:
            for table in tables:
                try:
                    cur = conn.execute(
                        f"UPDATE {table} SET station_id = ? WHERE station_id = ?",
                        (new, old),
                    )
                    n = int(getattr(cur, "rowcount", 0) or 0)
                    if n:
                        updated[f"{table}:{old}->{new}"] = n
                except Exception:
                    continue
    return updated


def load_forecast_verification_scores(
    cfg: dict | None = None,
    station_id: str | None = None,
) -> pd.DataFrame:
    init_db(cfg)
    cfg = cfg or load_config()
    with connect(cfg) as conn:
        if station_id:
            aliases = _station_id_aliases(str(station_id))
            placeholders = ", ".join("?" for _ in aliases)
            return pd.read_sql_query(
                f"SELECT * FROM forecast_verification_scores WHERE station_id IN ({placeholders}) "
                "ORDER BY channel, mae",
                conn,
                params=aliases,
            )
        return pd.read_sql_query(
            "SELECT * FROM forecast_verification_scores ORDER BY channel, mae", conn
        )


def replace_ensemble_forecast(
    frame: pd.DataFrame,
    cfg: dict | None = None,
    station_id: str | None = None,
) -> int:
    init_db(cfg)
    if frame.empty:
        return 0
    cfg = cfg or load_config()
    sid = str(station_id or cfg["station"]["id"])
    df = frame.copy()
    df["station_id"] = sid
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "bias_corrected" not in df.columns:
        df["bias_corrected"] = 0
    else:
        df["bias_corrected"] = df["bias_corrected"].fillna(False).astype(int)
    cols = [
        "timestamp", "station_id", "temperature_c", "humidity_pct", "pressure_hpa", "rainfall_mm",
        "wind_speed_ms", "wind_direction_deg",
        "temperature_c_spread", "humidity_pct_spread", "pressure_hpa_spread", "rainfall_mm_spread",
        "source", "fetched_at", "bias_corrected",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    # Turso rejects NaN floats; convert to SQL NULL.
    clean = df[cols].astype(object).where(pd.notnull(df[cols]), None)
    records = clean.to_dict(orient="records")
    with connect(cfg) as conn:
        conn.execute("DELETE FROM ensemble_forecast_hourly WHERE station_id = ?", (sid,))
        conn.executemany(
            f"INSERT INTO ensemble_forecast_hourly ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
            records,
        )
        return len(records)


def load_ensemble_forecast(cfg: dict | None = None, station_id: str | None = None) -> pd.DataFrame:
    init_db(cfg)
    cfg = cfg or load_config()
    sid = str(station_id or cfg["station"]["id"])
    with connect(cfg) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM ensemble_forecast_hourly WHERE station_id = ? ORDER BY timestamp",
            conn,
            params=(sid,),
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    return df


def replace_nwp_members(
    members: dict[str, pd.DataFrame],
    cfg: dict | None = None,
    station_id: str | None = None,
) -> int:
    init_db(cfg)
    cfg = cfg or load_config()
    sid = str(station_id or cfg["station"]["id"])
    total = 0
    with connect(cfg) as conn:
        conn.execute("DELETE FROM nwp_member_hourly WHERE station_id = ?", (sid,))
        for name, frame in members.items():
            df = frame.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            df["member"] = name
            df["station_id"] = sid
            if "fetched_at" in df.columns:
                df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            cols = [
                "timestamp", "station_id", "member", "temperature_c", "humidity_pct", "pressure_hpa",
                "rainfall_mm", "wind_speed_ms", "wind_direction_deg", "fetched_at",
            ]
            for col in cols:
                if col not in df.columns:
                    df[col] = None
            clean = df[cols].astype(object).where(pd.notnull(df[cols]), None)
            conn.executemany(
                f"INSERT INTO nwp_member_hourly ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})",
                clean.to_dict(orient="records"),
            )
            total += len(df)
    return total


def load_nwp_members(cfg: dict | None = None, station_id: str | None = None) -> pd.DataFrame:
    init_db(cfg)
    cfg = cfg or load_config()
    sid = str(station_id or cfg["station"]["id"])
    with connect(cfg) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM nwp_member_hourly WHERE station_id = ? ORDER BY timestamp, member",
            conn,
            params=(sid,),
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
