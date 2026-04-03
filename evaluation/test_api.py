"""API tests for the Task 1.4 session-centric stub endpoints."""

from __future__ import annotations

import json
import sqlite3
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import app, database, experiment_logger
from backend.mock_pipeline import count_syllables


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


def build_mp3_bytes(*, sample_rate: int = 22_050, duration_seconds: float = 1.0) -> bytes:
    timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    stereo = np.column_stack(
        (
            0.5 * np.sin(2 * np.pi * 220.0 * timeline),
            0.25 * np.sin(2 * np.pi * 440.0 * timeline),
        )
    ).astype(np.float32)
    output = Path("storage") / "test_tmp_api.mp3"
    sf.write(output, stereo, sample_rate, format="MP3")
    try:
        return output.read_bytes()
    finally:
        output.unlink(missing_ok=True)


def test_task_1_4_api_flow() -> None:
    reset_storage()
    client = TestClient(app)

    created = client.post("/session/create", json={"user_params": {"genre": "indie pop"}})
    assert created.status_code == 200
    session_state = created.json()["state"]
    session_id = session_state["session_id"]
    assert session_state["user_params"]["genre"] == "indie pop"

    sessions = client.get("/sessions")
    assert sessions.status_code == 200
    assert any(session["session_id"] == session_id for session in sessions.json())

    fetched = client.get(f"/session/{session_id}/state")
    assert fetched.status_code == 200
    assert fetched.json()["state"]["session_id"] == session_id

    melody = client.post(
        "/melody/from-hum",
        files={"audio": ("hum.wav", b"RIFF....WAVEfmt", "audio/wav")},
        data={"session_id": session_id, "prompt": "chorus", "mood": "uplift", "tempo_bpm": "120"},
    )
    assert melody.status_code == 200
    melody_body = melody.json()
    assert melody_body["session_id"] == session_id
    assert len(melody_body["melody_notes"]) >= 1
    assert melody_body["melody_profile"] is not None
    assert melody_body["detected_key"] == "C major"

    artifact_id = melody_body["artifacts"][0]["artifact_id"]
    listing = client.get(f"/artifact/{artifact_id}")
    assert listing.status_code == 200
    assert len(listing.json()["files"]) == 1

    chords = client.post(
        "/chords/from-lyrics",
        json={"session_id": session_id, "text": "Love under the sunrise"},
    )
    assert chords.status_code == 200
    assert len(chords.json()["chord_progressions"]) == 3

    recognise = client.post("/chords/from-melody", json={"session_id": session_id})
    assert recognise.status_code == 200
    recognise_body = recognise.json()
    assert recognise_body["session_id"] == session_id
    assert len(recognise_body["recognised_chords"]) == 3
    assert recognise_body["explanation_report"]["source_action"] == "chords_from_melody"

    continuation = client.post("/melody/continue", json={"session_id": session_id, "num_suggestions": 3})
    assert continuation.status_code == 200
    assert len(continuation.json()["melody_suggestions"]) == 3

    lyrics = client.post("/suggest/lyrics", json={"session_id": session_id, "mode": "continue"})
    assert lyrics.status_code == 200
    lyric_suggestions = lyrics.json()["lyric_suggestions"]
    assert len(lyric_suggestions) == 3
    assert lyric_suggestions[0]["syllable_count"] > len(lyric_suggestions[0]["text"].split())

    refined = client.patch(
        f"/session/{session_id}/refine",
        json={"instruction": "make it darker", "target": "chords"},
    )
    assert refined.status_code == 200
    refined_state = refined.json()["state"]
    assert refined_state["user_params"]["last_refinement"] == "make it darker"
    assert refined_state["user_params"]["last_refinement_target"] == "chords"
    assert "last_refinement_interpretation" in refined_state["user_params"]

    chat = client.post(
        f"/session/{session_id}/chat",
        json={"message": "Why this harmony?"},
    )
    assert chat.status_code == 200
    chat_body = chat.json()
    assert chat_body["session_id"] == session_id
    assert chat_body["reply"]["role"] == "assistant"
    assert len(chat_body["chat_history"]) == 2

    final_state = client.get(f"/session/{session_id}/state")
    assert final_state.status_code == 200
    body = final_state.json()["state"]
    assert len(body["recognised_chords"]) == 3
    assert len(body["melody_suggestions"]) == 3
    assert len(body["lyric_suggestions"]) == 3
    assert len(body["chat_history"]) == 2
    assert body["explanation_report"]["source_action"] == "session_chat"
    assert len(body["history"]) >= 7


def test_melody_from_hum_uses_mock_pipeline_for_melancholic_key() -> None:
    reset_storage()
    client = TestClient(app)

    response = client.post(
        "/melody/from-hum",
        files={"audio": ("hum.wav", b"RIFF....WAVEfmt", "audio/wav")},
        data={"prompt": "verse", "mood": "melancholic", "tempo_bpm": "96"},
    )

    assert response.status_code == 200
    assert response.json()["detected_key"] == "A minor"


def test_count_syllables_counts_syllables_not_words() -> None:
    assert count_syllables("beneath") == 2


def test_melody_from_hum_accepts_mp3_upload() -> None:
    reset_storage()
    client = TestClient(app)

    response = client.post(
        "/melody/from-hum",
        files={"audio": ("hum.mp3", build_mp3_bytes(), "audio/mpeg")},
        data={"prompt": "hook", "mood": "uplift", "tempo_bpm": "118"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["melody_notes"]) >= 1
    assert body["melody_profile"] is not None


def test_request_logging_emits_structured_json(caplog) -> None:
    reset_storage()
    client = TestClient(app)

    created = client.post("/session/create", json={"user_params": {"genre": "dream pop"}})
    assert created.status_code == 200
    session_id = created.json()["state"]["session_id"]

    logs = [json.loads(record.message) for record in caplog.records if record.name == "backend.api"]
    assert logs

    entry = next(log for log in logs if log["event"] == "request_completed" and log["endpoint"] == "/session/create")
    assert entry["session_id"] == session_id
    assert entry["pipeline_stage"] == "session"
    assert entry["use_case"] is None
    assert entry["success"] is True
    assert entry["error"] is None
    assert isinstance(entry["latency_ms"], float)
