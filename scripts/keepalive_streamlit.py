#!/usr/bin/env python
"""Wake / keep a Streamlit Community Cloud app from sleeping.

Requires a real browser visit (Playwright). HTTP GET alone is not enough.
"""

from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://heimdall-eye.streamlit.app/"
WAKE_LABEL = "Yes, get this app back up!"


def main() -> None:
    url = (os.environ.get("STREAMLIT_APP_URL") or DEFAULT_URL).strip()
    print(f"Visiting {url}", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(5)

        wake = page.get_by_role("button", name=WAKE_LABEL)
        if wake.count() > 0:
            print("App was asleep — clicking wake button.", flush=True)
            wake.first.click()
            time.sleep(45)
        else:
            print("App already awake (no sleep button).", flush=True)
            time.sleep(8)

        # Confirm Streamlit core is up when possible
        health_url = url.rstrip("/") + "/_stcore/health"
        try:
            response = page.request.get(health_url, timeout=30_000)
            print(f"Health {response.status}: {response.text()[:120]}", flush=True)
        except Exception as exc:
            print(f"Health check skipped: {exc}", flush=True)

        browser.close()

    print("Keepalive done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Keepalive failed: {exc}", flush=True)
        raise SystemExit(1) from exc
