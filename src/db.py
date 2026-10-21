"""Database connection: local SQLite by default, Turso/libSQL when credentials exist."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.config import load_config


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    path = Path(cfg["storage"]["sqlite_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        raw = secrets.get(name)  # type: ignore[union-attr]
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None
    except Exception:
        return None


def remote_credentials() -> tuple[str, str] | None:
    """Return (url, token) when Turso/libSQL remote mode is configured."""
    url = _secret("TURSO_DATABASE_URL") or _secret("LIBSQL_URL")
    token = _secret("TURSO_AUTH_TOKEN") or _secret("LIBSQL_AUTH_TOKEN")
    if url and token:
        return url, token
    return None


def using_remote_db() -> bool:
    return remote_credentials() is not None


def _open_connection(cfg: dict | None = None) -> Any:
    creds = remote_credentials()
    if creds is not None:
        import libsql

        url, token = creds
        conn = libsql.connect(database=url, auth_token=token)
        return conn

    conn = sqlite3.connect(db_path(cfg))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect(cfg: dict | None = None) -> Iterator[Any]:
    conn = _open_connection(cfg)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
