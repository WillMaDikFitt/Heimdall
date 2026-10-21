#!/usr/bin/env python
"""Copy weather prediction history from data/weather.db into data/heimdall.db.

Remaps WS-LESHKA -> WS-INDRA and WS-CARDAMOM -> WS-ISSA.
Does not overwrite Heimdall telemetry or freshly simulated weather readings.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.store import init_db
from src.paths import PROJECT_ROOT

ID_MAP = {
    "WS-LESHKA": "WS-INDRA",
    "WS-CARDAMOM": "WS-ISSA",
    "WS-INDRA": "WS-INDRA",
    "WS-ISSA": "WS-ISSA",
}

# Prediction / skill tables we want back under Heimdall.
STATION_TABLES = [
    "ensemble_forecast_hourly",
    "nwp_member_hourly",
    "forecast_verification_scores",
    "calibrated_readings",
    "calibration_coefficients",
]

# Shared site-agnostic tables (copy when destination is empty).
GLOBAL_TABLES = [
    "imd_daily_rain",
    "power_readings",
    "source_validation_scores",
    "forecast_scores",
    "forecast_predictions",
    "walkforward_scores",
]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _remap_station(row: sqlite3.Row, columns: list[str]) -> dict | None:
    data = {col: row[col] for col in columns}
    if "station_id" in data and data["station_id"] is not None:
        old = str(data["station_id"])
        if old not in ID_MAP:
            return None
        data["station_id"] = ID_MAP[old]
    return data


def _copy_station_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str) -> int:
    if table not in {
        r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }:
        print(f"  skip {table}: missing in weather.db")
        return 0
    cols = _columns(src, table)
    if "station_id" not in cols:
        print(f"  skip {table}: no station_id")
        return 0
    dst_cols = set(_columns(dst, table))
    use_cols = [c for c in cols if c in dst_cols]
    rows = src.execute(f"SELECT * FROM {table}").fetchall()
    payload = []
    for row in rows:
        mapped = _remap_station(row, cols)
        if mapped is None:
            continue
        payload.append({c: mapped[c] for c in use_cols})
    if not payload:
        print(f"  {table}: nothing to copy")
        return 0

    # Replace only the remapped station ids so INDRA/ISSA prediction history is restored.
    targets = sorted({row["station_id"] for row in payload})
    for sid in targets:
        dst.execute(f"DELETE FROM {table} WHERE station_id = ?", (sid,))
    placeholders = ", ".join(":" + c for c in use_cols)
    col_sql = ", ".join(use_cols)
    dst.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})",
        payload,
    )
    print(f"  {table}: copied {len(payload)} rows -> {targets}")
    return len(payload)


def _copy_global_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str) -> int:
    tables_src = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    tables_dst = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if table not in tables_src or table not in tables_dst:
        print(f"  skip {table}: missing")
        return 0
    existing = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if existing:
        print(f"  keep {table}: destination already has {existing} rows")
        return 0
    cols_src = _columns(src, table)
    cols_dst = set(_columns(dst, table))
    use_cols = [c for c in cols_src if c in cols_dst]
    rows = src.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        print(f"  {table}: empty source")
        return 0
    payload = [{c: row[c] for c in use_cols} for row in rows]
    placeholders = ", ".join(":" + c for c in use_cols)
    col_sql = ", ".join(use_cols)
    dst.executemany(
        f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})",
        payload,
    )
    print(f"  {table}: copied {len(payload)} rows")
    return len(payload)


def main() -> None:
    src_path = PROJECT_ROOT / "data" / "weather.db"
    dst_path = PROJECT_ROOT / "data" / "heimdall.db"
    if not src_path.exists():
        raise SystemExit(f"Missing source database: {src_path}")
    init_db()  # ensure heimdall schema exists
    src = sqlite3.connect(src_path)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(dst_path)
    dst.row_factory = sqlite3.Row
    print(f"Migrating {src_path.name} -> {dst_path.name}")
    total = 0
    for table in STATION_TABLES:
        total += _copy_station_table(src, dst, table)
    for table in GLOBAL_TABLES:
        total += _copy_global_table(src, dst, table)
    dst.commit()
    src.close()
    dst.close()
    print(f"Done. Migrated {total} row writes.")
    print("Original data/weather.db left untouched as backup.")


if __name__ == "__main__":
    main()
