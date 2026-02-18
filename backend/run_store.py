"""SQLite store for run metadata and artifact links."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class RunStore:
    def __init__(self, db_path: str | Path = "storage/runs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  module_name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  duration_ms REAL,
                  error TEXT,
                  metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  artifact_id TEXT NOT NULL,
                  artifact_type TEXT NOT NULL,
                  relative_path TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )

    def start_run(self, module_name: str, metadata: dict[str, Any] | None = None) -> str:
        run_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, module_name, status, started_at, metadata_json)
                VALUES(?, ?, 'running', ?, ?)
                """,
                (run_id, module_name, _now_iso(), json.dumps(metadata or {})),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "finished",
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, duration_ms = ?, error = ?
                WHERE run_id = ?
                """,
                (status, _now_iso(), duration_ms, error, run_id),
            )

    def link_artifact(
        self,
        run_id: str,
        artifact_id: str,
        artifact_type: str,
        relative_path: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(run_id, artifact_id, artifact_type, relative_path, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (run_id, artifact_id, artifact_type, relative_path, _now_iso()),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, artifact_id, artifact_type, relative_path, created_at FROM artifacts WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

