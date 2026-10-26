"""Database connection: local SQLite by default, Turso/libSQL when credentials exist."""

from __future__ import annotations

import math
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.config import load_config
from src.paths import PROJECT_ROOT

_NAMED_PARAM = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
_DOTENV_LOADED = False


def _load_dotenv() -> None:
    """Load project .env into os.environ once (local CLI / Streamlit)."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except Exception:
        return


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    path = Path(cfg["storage"]["sqlite_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _secret(name: str) -> str | None:
    _load_dotenv()
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


class _IterableCursor:
    """Make libsql cursors behave like sqlite3 (iterable + fetch*)."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor
        self._rows: list[Any] | None = None
        self._index = 0

    def _materialize(self) -> list[Any]:
        if self._rows is None:
            self._rows = list(self._cursor.fetchall())
            self._index = 0
        return self._rows

    def __iter__(self):
        return iter(self._materialize())

    def fetchone(self) -> Any:
        rows = self._materialize()
        if self._index >= len(rows):
            return None
        row = rows[self._index]
        self._index += 1
        return row

    def fetchall(self) -> list[Any]:
        rows = self._materialize()
        leftover = rows[self._index :]
        self._index = len(rows)
        return leftover

    def fetchmany(self, size: int | None = None) -> list[Any]:
        rows = self._materialize()
        if size is None:
            size = 1
        chunk = rows[self._index : self._index + size]
        self._index += len(chunk)
        return chunk

    @property
    def description(self) -> Any:
        return getattr(self._cursor, "description", None)

    @property
    def rowcount(self) -> Any:
        return getattr(self._cursor, "rowcount", -1)

    def close(self) -> None:
        close = getattr(self._cursor, "close", None)
        if callable(close):
            close()


def _sanitize_param(value: Any) -> Any:
    """Turso/Hrana rejects NaN/Inf (JSON null) for f64 binds — store SQL NULL instead."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        # pandas / numpy NA
        if value is not None and value != value:  # noqa: PLR0124
            return None
    except Exception:
        pass
    return value


def _sanitize_params(parameters: Any) -> Any:
    if isinstance(parameters, dict):
        return {key: _sanitize_param(val) for key, val in parameters.items()}
    if isinstance(parameters, (list, tuple)):
        return type(parameters)(_sanitize_param(val) for val in parameters)
    return _sanitize_param(parameters)


def _to_qmark(sql: str, parameters: Any) -> tuple[str, Any]:
    """Convert sqlite3-style :name dict params to libsql-compatible qmark tuples."""
    parameters = _sanitize_params(parameters)
    if not isinstance(parameters, dict):
        return sql, parameters
    values: list[Any] = []

    def repl(match: re.Match[str]) -> str:
        values.append(parameters[match.group(1)])
        return "?"

    return _NAMED_PARAM.sub(repl, sql), values


class _CompatConnection:
    """Thin wrapper so remote libsql matches sqlite3 connection habits."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, parameters: Any = ()) -> _IterableCursor:
        if parameters in ((), [], None):
            cursor = self._conn.execute(sql)
        else:
            sql, parameters = _to_qmark(sql, parameters)
            cursor = self._conn.execute(sql, parameters)
        return _IterableCursor(cursor)

    def executemany(self, sql: str, seq_of_parameters: Any) -> _IterableCursor:
        seq = list(seq_of_parameters or [])
        if not seq:
            return _IterableCursor(self._conn.execute(sql))
        # Prefer batched qmark tuples; keep batches modest for Turso HTTP limits.
        batch_size = 100
        qsql = sql
        converted: list[Any] = []
        if isinstance(seq[0], dict):
            for item in seq:
                qsql, values = _to_qmark(sql, item)
                converted.append(values)
        else:
            converted = [_sanitize_params(item) for item in seq]
        cursor = None
        for start in range(0, len(converted), batch_size):
            cursor = self._conn.executemany(qsql, converted[start : start + batch_size])
        return _IterableCursor(cursor)

    def executescript(self, script: str) -> Any:
        return self._conn.executescript(script)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        rollback = getattr(self._conn, "rollback", None)
        if callable(rollback):
            rollback()

    def close(self) -> None:
        self._conn.close()

    def cursor(self) -> Any:
        return self._conn.cursor()


def _open_connection(cfg: dict | None = None) -> Any:
    creds = remote_credentials()
    if creds is not None:
        import libsql

        url, token = creds
        return _CompatConnection(libsql.connect(database=url, auth_token=token))

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
