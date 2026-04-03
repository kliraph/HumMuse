"""Smoke checks for EPIC 2 FastAPI endpoints."""

from __future__ import annotations

import sqlite3
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app, database, experiment_logger


def reset_storage() -> None:
    storage = Path("storage")
    storage.mkdir(parents=True, exist_ok=True)
    database._init_db()
    experiment_logger.run_store._init_db()
    if experiment_logger.artifact_store.root_dir.exists():
        shutil.rmtree(experiment_logger.artifact_store.root_dir, ignore_errors=True)
    experiment_logger.artifact_store.root_dir.mkdir(parents=True, exist_ok=True)

    for db_path in (database.db_path, experiment_logger.run_store.db_path):
        with sqlite3.connect(db_path) as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")


def run() -> None:
    reset_storage()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    version = client.get("/version")
    assert version.status_code == 200
    assert "version" in version.json()

    melody = client.post(
        "/melody/from-hum",
        files={"audio": ("hum.wav", b"RIFF....WAVEfmt", "audio/wav")},
        data={"prompt": "chorus", "mood": "uplift", "tempo_bpm": "120"},
    )
    assert melody.status_code == 200
    body = melody.json()
    assert "run_id" in body
    assert len(body["artifacts"]) >= 1

    midi_url = next(a["url"] for a in body["artifacts"] if a["kind"] == "midi")
    midi_resp = client.get(midi_url)
    assert midi_resp.status_code == 200
    assert len(midi_resp.content) > 0

    chords = client.post("/chords/from-lyrics", json={"text": "Love under the sunrise"})
    assert chords.status_code == 200
    assert len(chords.json()["top_progressions"]) == 3

    created = client.post("/session/create")
    assert created.status_code == 200
    session_id = created.json()["state"]["session_id"]

    melody_session = client.post(
        "/melody/from-hum",
        files={"audio": ("hum.wav", b"RIFF....WAVEfmt", "audio/wav")},
        data={"session_id": session_id, "prompt": "verse", "mood": "uplift", "tempo_bpm": "100"},
    )
    assert melody_session.status_code == 200

    recognised = client.post("/chords/from-melody", json={"session_id": session_id})
    assert recognised.status_code == 200
    assert len(recognised.json()["recognised_chords"]) == 3

    chat = client.post(f"/session/{session_id}/chat", json={"message": "Why this harmony?"})
    assert chat.status_code == 200
    assert chat.json()["reply"]["role"] == "assistant"

    wb = client.post("/writerblock/help", json={"text": "I cannot finish this verse", "mood": "tense"})
    assert wb.status_code == 200
    assert len(wb.json()["suggestions"]) >= 2

    print("EPIC 2 smoke checks passed")


if __name__ == "__main__":
    run()
