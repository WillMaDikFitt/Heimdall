#!/usr/bin/env python
"""Upload local data/heimdall.db into a Turso/libSQL database.

Requires:
  TURSO_DATABASE_URL
  TURSO_AUTH_TOKEN

Create a free DB at https://turso.tech then:
  turso db show --url heimdall
  turso db tokens create heimdall
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import libsql

from src.config import load_config
from src.db import db_path, remote_credentials
from src.ingestion.store import init_db

SKIP_TABLES = {"sqlite_sequence"}


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

    # Ensure remote schema exists (init_db uses remote when creds are set).
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

    dst = libsql.connect(database=url, auth_token=token)
    for table in tables:
        cols = [row[1] for row in src.execute(f"PRAGMA table_info({table})")]
        if not cols:
            continue
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        print(f"  {table}: {len(rows)} rows", flush=True)
        if not rows:
            continue
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(cols)
        # Replace table contents so re-runs are idempotent.
        dst.execute(f"DELETE FROM {table}")
        payload = [tuple(row[c] for c in cols) for row in rows]
        dst.executemany(
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
            payload,
        )
        dst.commit()

    src.close()
    dst.close()
    print("Upload complete.", flush=True)


if __name__ == "__main__":
    # Allow dotenv-style local export without requiring python-dotenv.
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    main()
