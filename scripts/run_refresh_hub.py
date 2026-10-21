#!/usr/bin/env python
"""Refresh live Overview outlook and archive it for later rain verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.hub_refresh import run_hub_refresh


def main() -> None:
    cfg = load_config()
    result = run_hub_refresh(cfg)
    rain_steps = [s for s in result.get("steps", []) if s.get("name") == "verify"]
    if rain_steps:
        print(rain_steps[0]["detail"], flush=True)
    if not result.get("ok"):
        raise SystemExit(result.get("error") or "Hub refresh failed.")


if __name__ == "__main__":
    main()
