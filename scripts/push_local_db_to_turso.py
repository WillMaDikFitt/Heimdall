#!/usr/bin/env python
"""Upload local data/heimdall.db into a Turso/libSQL database.

Uses multi-row INSERT statements (faster over HTTP than executemany).
Requires TURSO_DATABASE_URL + TURSO_AUTH_TOKEN (env or .env).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import libsql

from src.config import load_config
from src.db import db_path, remote_credentials
from src.ingestion.store import init_db

SKIP_TABLES = {"sqlite_sequence"}
BATCH_SIZE = 80


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:  # NaN
            return "NULL"
        return repr(float(value)) if isinstance(value, float) else str(int(value))
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _insert_batches(dst: object, table: str, cols: list[str], rows: list) -> None:
    col_list = ", ".join(cols)
    total = len(rows)
    for start in range(0, total, BATCH_SIZE):
        chunk = rows[start : start + BATCH_SIZE]
        values_sql = ", ".join(
            "(" + ", ".join(_sql_literal(row[c]) for c in cols) + ")" for row in chunk
        )
        sql = f"INSERT INTO {table} ({col_list}) VALUES {values_sql}"
        for attempt in range(1, 5):
            try:
                dst.execute(sql)  # type: ignore[attr-defined]
                dst.commit()  # type: ignore[attr-defined]
                break
            except Exception as exc:
                if attempt >= 4:
                    raise
                print(f"    retry {attempt}: {exc}", flush=True)
                time.sleep(1.2 * attempt)
        done = min(start + BATCH_SIZE, total)
        if done == total or done % (BATCH_SIZE * 5) == 0 or start == 0:
            print(f"    … {done}/{total}", flush=True)


def main() -> None:
    creds = remote_credentials()
    if creds is None:
        raise SystemExit(
            "Set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in the environment first."
        )
    url, token = creds
    local = db_path(load_config())
    if not local.exists():
        raise SystemExit(f"Local database not found: {local}")

    print(f"Local DB: {local} ({local.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"Remote:   {url}", flush=True)
    print("Creating schema on Turso…", flush=True)
    init_db()

    src = sqlite3.connect(local)
    src.row_factory = sqlite3.Row
    tables = [
        row[0]
        for row in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        if row[0] not in SKIP_TABLES and not str(row[0]).startswith("sqlite_")
    ]

    # Upload smaller tables first so Progress/Forecast work even if a large
    # telemetry table is still transferring.
    def sort_key(name: str) -> tuple[int, str]:
        priority = {
            "ensemble_forecast_hourly": 0,
            "nwp_member_hourly": 1,
            "forecast_verification_scores": 2,
            "reference_readings": 3,
            "readings": 4,
            "calibrated_readings": 5,
            "calibration_coefficients": 6,
            "telemetry": 99,
        }
        return (priority.get(name, 50), name)

    tables = sorted(tables, key=sort_key)

    dst = libsql.connect(database=url, auth_token=token)
    for table in tables:
        cols = [row[1] for row in src.execute(f"PRAGMA table_info({table})")]
        if not cols:
            continue
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        print(f"  {table}: {len(rows)} rows", flush=True)
        dst.execute(f"DELETE FROM {table}")
        dst.commit()
        if not rows:
            continue
        _insert_batches(dst, table, cols, rows)

    src.close()
    dst.close()
    print("Upload complete.", flush=True)


if __name__ == "__main__":
    _load_dotenv()
    main()
