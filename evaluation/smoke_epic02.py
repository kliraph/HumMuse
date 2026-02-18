"""Smoke checks for EPIC 2 FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app


def run() -> None:
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

    wb = client.post("/writerblock/help", json={"text": "I cannot finish this verse", "mood": "tense"})
    assert wb.status_code == 200
    assert len(wb.json()["suggestions"]) >= 2

    print("EPIC 2 smoke checks passed")


if __name__ == "__main__":
    run()

