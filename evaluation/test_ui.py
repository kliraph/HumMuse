"""Lightweight checks for the Task 2.1 Streamlit shell."""

from __future__ import annotations

from ui.audio_utils import decode_midi_bytes, render_audio_from_session, synthesize_wave_from_notes
from ui.app import (
    DEFAULT_API_URL,
    accept_melody_suggestion,
    artifact_by_kind,
    build_audio_file_tuple,
    confidence_to_color,
    extract_pipeline_timings,
    format_midi_pitch,
    format_session_option,
    note_rows,
    numbered_label,
    progression_caption,
    progression_title,
    refine_session,
    send_chat_message,
)


class DummyUpload:
    def __init__(self, name: str, payload: bytes, mime_type: str) -> None:
        self.name = name
        self._payload = payload
        self.type = mime_type

    def getvalue(self) -> bytes:
        return self._payload


def test_ui_helpers_expose_expected_defaults() -> None:
    assert DEFAULT_API_URL == "http://127.0.0.1:8000"


def test_format_session_option_includes_id_and_timestamp() -> None:
    label = format_session_option(
        {
            "session_id": "1234-session",
            "updated_at": "2026-03-20T18:30:00Z",
        }
    )
    assert "1234-session" in label
    assert "2026-03-20 18:30:00 UTC" in label


def test_build_audio_file_tuple_preserves_metadata_and_bytes() -> None:
    uploaded = DummyUpload("idea.wav", b"RIFFDATA", "audio/wav")
    filename, payload, mime_type = build_audio_file_tuple(uploaded)

    assert filename == "idea.wav"
    assert payload == b"RIFFDATA"
    assert mime_type == "audio/wav"


def test_note_rows_keeps_expected_note_columns() -> None:
    rows = note_rows(
        [
            {
                "pitch": 64,
                "onset": 0.0,
                "duration": 1.0,
                "velocity": 96,
                "confidence": 0.91,
                "extra": "ignored",
            }
        ]
    )

    assert rows == [
        {
            "pitch": 64,
            "pitch_label": "E4 (64)",
            "onset": 0.0,
            "duration": 1.0,
            "velocity": 96,
            "confidence": 0.91,
        }
    ]


def test_format_midi_pitch_returns_note_name_and_number() -> None:
    assert format_midi_pitch(60) == "C4 (60)"
    assert format_midi_pitch(69) == "A4 (69)"


def test_artifact_by_kind_returns_matching_artifact() -> None:
    artifacts = [
        {"kind": "midi", "url": "http://localhost/melody.mid"},
        {"kind": "timings", "url": "http://localhost/timings.json"},
    ]

    assert artifact_by_kind(artifacts, "timings") == {"kind": "timings", "url": "http://localhost/timings.json"}
    assert artifact_by_kind(artifacts, "missing") is None


def test_extract_pipeline_timings_fetches_timing_artifact(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_api_get_json_by_url(url: str) -> dict[str, object]:
        captured["url"] = url
        return {"step_latency_ms": {"decode_audio": 1.2}}

    monkeypatch.setattr("ui.app.api_get_json_by_url", fake_api_get_json_by_url)

    result = extract_pipeline_timings(
        {
            "artifacts": [
                {"kind": "timings", "url": "http://localhost/artifact/timings.json"},
            ]
        }
    )

    assert result == {"step_latency_ms": {"decode_audio": 1.2}}
    assert captured["url"] == "http://localhost/artifact/timings.json"


def test_progression_helpers_format_cards_cleanly() -> None:
    progression = {
        "chords": ["Am", "F", "C", "G"],
        "score": 0.88,
        "harmonic_function": "predominant lift",
        "explanation": "Stable pop cadence with a reflective color.",
    }

    assert progression_title(progression) == "Am -> F -> C -> G"
    assert progression_caption(progression) == "Model confidence: 0.88 | predominant lift | Stable pop cadence with a reflective color."


def test_numbered_label_formats_options_for_suggestions() -> None:
    assert numbered_label(2, "Continue upward and resolve") == "2. Continue upward and resolve"


def test_refine_session_uses_expected_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_api_patch(api_base_url: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        captured["api_base_url"] = api_base_url
        captured["path"] = path
        captured["payload"] = payload
        return {"state": {"session_id": "abc"}}

    monkeypatch.setattr("ui.app.api_patch", fake_api_patch)

    result = refine_session(
        "http://127.0.0.1:8000",
        session_id="session-123",
        instruction="make it jazzier",
        target="chords",
    )

    assert result == {"state": {"session_id": "abc"}}
    assert captured == {
        "api_base_url": "http://127.0.0.1:8000",
        "path": "/session/session-123/refine",
        "payload": {"instruction": "make it jazzier", "target": "chords"},
    }


def test_send_chat_message_uses_expected_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_api_post(api_base_url: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        captured["api_base_url"] = api_base_url
        captured["path"] = path
        captured["payload"] = payload
        return {"reply": {"role": "assistant", "content": "Grounded reply"}}

    monkeypatch.setattr("ui.app.api_post", fake_api_post)

    result = send_chat_message(
        "http://127.0.0.1:8000",
        session_id="session-123",
        message="Why Dm?",
    )

    assert result == {"reply": {"role": "assistant", "content": "Grounded reply"}}
    assert captured == {
        "api_base_url": "http://127.0.0.1:8000",
        "path": "/session/session-123/chat",
        "payload": {"message": "Why Dm?"},
    }


def test_accept_melody_suggestion_uses_expected_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_api_post(api_base_url: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        captured["api_base_url"] = api_base_url
        captured["path"] = path
        captured["payload"] = payload
        return {"state": {"session_id": "session-123"}}

    monkeypatch.setattr("ui.app.api_post", fake_api_post)

    result = accept_melody_suggestion(
        "http://127.0.0.1:8000",
        session_id="session-123",
        suggestion_index=2,
    )

    assert result == {"state": {"session_id": "session-123"}}
    assert captured == {
        "api_base_url": "http://127.0.0.1:8000",
        "path": "/melody/accept",
        "payload": {"session_id": "session-123", "suggestion_index": 2},
    }


def test_confidence_to_color_maps_expected_bands() -> None:
    assert confidence_to_color(0.9) == "#d9f99d"
    assert confidence_to_color(0.75) == "#fde68a"
    assert confidence_to_color(0.4) == "#fecaca"


def test_decode_midi_bytes_handles_base64_text() -> None:
    assert decode_midi_bytes("TVRoZA==") == b"MThd"


def test_synthesize_wave_from_notes_returns_wav_bytes() -> None:
    audio = synthesize_wave_from_notes(
        [{"pitch": 60, "onset": 0.0, "duration": 0.25, "velocity": 96, "confidence": 0.9}]
    )

    assert audio[:4] == b"RIFF"
    assert b"WAVE" in audio[:16]


def test_render_audio_from_session_falls_back_to_note_synth() -> None:
    audio, source = render_audio_from_session(
        {
            "melody_midi": "TVRoZA==",
            "melody_notes": [
                {"pitch": 60, "onset": 0.0, "duration": 0.25, "velocity": 96, "confidence": 0.9}
            ],
        }
    )

    assert source == "note_synth"
    assert audio[:4] == b"RIFF"
