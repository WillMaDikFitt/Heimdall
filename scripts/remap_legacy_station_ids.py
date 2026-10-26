#!/usr/bin/env python
"""Rewrite WS-LESHKA/WS-CARDAMOM rows to WS-INDRA/WS-ISSA in the active DB."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db import using_remote_db
from src.ingestion.store import remap_legacy_station_ids


def main() -> None:
    print("Target:", "Turso" if using_remote_db() else "local SQLite", flush=True)
    updated = remap_legacy_station_ids()
    if not updated:
        print("No legacy station rows found.", flush=True)
        return
    for key, count in updated.items():
        print(f"  {key}: {count}", flush=True)
    print("Remap complete.", flush=True)


if __name__ == "__main__":
    main()
