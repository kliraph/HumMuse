"""SQLite bootstrap and helpers for session-centric backend state."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, db_path: str | Path = "storage/hummuse.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  state_json TEXT NOT NULL,
                  pipeline_version TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                  prompt_hash TEXT NOT NULL,
                  use_case TEXT NOT NULL,
                  response_json TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  model_version TEXT NOT NULL,
                  PRIMARY KEY(prompt_hash, model_version, use_case)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  pipeline_version TEXT NOT NULL,
                  metric_name TEXT NOT NULL,
                  value REAL NOT NULL,
                  timestamp TEXT NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            self._migrate_api_cache_table(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decoding_traces (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  pipeline_stage TEXT NOT NULL,
                  trace_json TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )

    def _migrate_api_cache_table(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api_cache)").fetchall()
        }
        if not columns:
            return

        if columns == {"prompt_hash", "use_case", "response_json", "timestamp", "model_version"}:
            return

        conn.execute("ALTER TABLE api_cache RENAME TO api_cache_legacy")
        conn.execute(
            """
            CREATE TABLE api_cache (
              prompt_hash TEXT NOT NULL,
              use_case TEXT NOT NULL,
              response_json TEXT NOT NULL,
              timestamp TEXT NOT NULL,
              model_version TEXT NOT NULL,
              PRIMARY KEY(prompt_hash, model_version, use_case)
            )
            """
        )
        if "prompt_hash" in columns and "response_json" in columns and "timestamp" in columns and "model_version" in columns:
            conn.execute(
                """
                INSERT INTO api_cache(prompt_hash, use_case, response_json, timestamp, model_version)
                SELECT prompt_hash, 'legacy', response_json, timestamp, model_version
                FROM api_cache_legacy
                """
            )
        conn.execute("DROP TABLE api_cache_legacy")

    def upsert_session(
        self,
        session_id: str,
        state_json: str,
        pipeline_version: str,
        *,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        created_value = created_at or _now_iso()
        updated_value = updated_at or _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(id, created_at, updated_at, state_json, pipeline_version)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  updated_at = excluded.updated_at,
                  state_json = excluded.state_json,
                  pipeline_version = excluded.pipeline_version
                """
                ,
                (session_id, created_value, updated_value, state_json, pipeline_version),
            )

    def get_session_row(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_session_rows(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, created_at, updated_at, pipeline_version
                FROM sessions
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def cache_response(self, prompt_hash: str, response_json: str, model_version: str, use_case: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO api_cache(prompt_hash, use_case, response_json, timestamp, model_version)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(prompt_hash, model_version, use_case) DO UPDATE SET
                  response_json = excluded.response_json,
                  timestamp = excluded.timestamp
                """
                ,
                (prompt_hash, use_case, response_json, _now_iso(), model_version),
            )

    def get_cached_response(self, prompt_hash: str, model_version: str, use_case: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT response_json
                FROM api_cache
                WHERE prompt_hash = ? AND model_version = ? AND use_case = ?
                """,
                (prompt_hash, model_version, use_case),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def record_decoding_trace(self, session_id: str, pipeline_stage: str, trace_json: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decoding_traces(session_id, pipeline_stage, trace_json, timestamp)
                VALUES(?, ?, ?, ?)
                """
                ,
                (session_id, pipeline_stage, trace_json, _now_iso()),
            )

    def record_metric(
        self,
        session_id: str,
        pipeline_version: str,
        metric_name: str,
        value: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics(session_id, pipeline_version, metric_name, value, timestamp)
                VALUES(?, ?, ?, ?, ?)
                """
                ,
                (session_id, pipeline_version, metric_name, value, _now_iso()),
            )
