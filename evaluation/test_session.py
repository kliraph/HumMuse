"""Persistence tests for Task 1.2-1.3 session storage."""

from __future__ import annotations

import sqlite3
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from backend.database import Database
from backend.session import SessionManager, SessionNotFoundError
from shared.schemas import (
    Action,
    ChatMessage,
    ExplanationReport,
    EmotionVector,
    LyricSuggestion,
    MelodyProfile,
    MelodySuggestion,
    NoteEvent,
    Progression,
    SessionState,
)


@pytest.fixture
def local_tmp_path() -> Path:
    path = Path("storage/test_tmp") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _build_state(session_id: str) -> SessionState:
    note = NoteEvent(pitch=64, onset=0.0, duration=1.0, velocity=96, confidence=0.98)
    return SessionState(
        session_id=session_id,
        melody_midi=b"\x80\x81\xffMThd\x00\x00",
        melody_notes=[note],
        melody_profile=MelodyProfile(
            interval_histogram=[0.1, 0.3, 0.6],
            rhythmic_density=1.5,
            pitch_range=(60, 72),
            contour="arch",
        ),
        detected_key="C major",
        detected_tempo=118.0,
        chord_progressions=[
            Progression(
                chords=["C", "G", "Am", "F"],
                score=0.91,
                harmonic_function="tonic to dominant cycle",
                explanation="Matches the lyric mood and implied cadence.",
            )
        ],
        lyrics_text="Hold the night until the morning sings",
        emotion_vector=EmotionVector(valence=0.72, arousal=0.44),
        melody_suggestions=[
            MelodySuggestion(
                midi_bytes=b"\x00\xfe\x7f",
                notes=[note],
                explanation="Continues the contour while staying in range.",
                coherence_score=0.87,
            )
        ],
        lyric_suggestions=[
            LyricSuggestion(
                text="Let the skyline learn our names",
                mode="continue",
                syllable_count=7,
            )
        ],
        explanation_report=ExplanationReport(
            source_action="test_seed",
            summary="Test explanation report",
            melody_confidence=[{"pitch": 64, "confidence": 0.98}],
            constraint_logs=[{"candidate": 1, "status": "accepted"}],
            chord_theory=[{"progression": ["C", "G", "Am", "F"]}],
            emotion_mapping={"valence": 0.72, "arousal": 0.44},
            cache_status={"use_case": "lyric", "cache_hit": False},
        ),
        chat_history=[
            ChatMessage(
                role="user",
                content="Why did you pick Am?",
                timestamp="2026-03-20T00:01:00Z",
            ),
            ChatMessage(
                role="assistant",
                content="Because A and C are prominent melody tones.",
                timestamp="2026-03-20T00:01:02Z",
            ),
        ],
        user_params={"genre": "indie pop", "complexity": "medium"},
        history=[
            Action(
                kind="session_started",
                payload={"source": "test"},
                timestamp="2026-03-20T00:00:00Z",
                source="backend",
            )
        ],
    )


def test_database_initializes_expected_tables(local_tmp_path: Path) -> None:
    database = Database(local_tmp_path / "hummuse.db")

    assert database.db_path.exists()

    with sqlite3.connect(database.db_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('sessions', 'api_cache', 'metrics')
            """
        ).fetchall()

    assert {row[0] for row in rows} == {"sessions", "api_cache", "metrics"}


def test_session_manager_round_trip_preserves_all_fields(local_tmp_path: Path) -> None:
    manager = SessionManager(Database(local_tmp_path / "hummuse.db"))

    created = manager.create_session()
    fetched = manager.get_session(created.session_id)
    assert fetched == created

    updated = _build_state(str(created.session_id))
    manager.update_session(created.session_id, updated)
    fetched_updated = manager.get_session(created.session_id)

    assert fetched_updated == updated
    assert fetched_updated.melody_midi == updated.melody_midi
    assert fetched_updated.melody_suggestions[0].midi_bytes == updated.melody_suggestions[0].midi_bytes
    assert fetched_updated.chat_history[0].content == "Why did you pick Am?"
    assert fetched_updated.explanation_report is not None


def test_list_sessions_returns_summary_metadata(local_tmp_path: Path) -> None:
    manager = SessionManager(Database(local_tmp_path / "hummuse.db"))

    created = manager.create_session()
    summaries = manager.list_sessions()

    assert len(summaries) == 1
    assert summaries[0].session_id == created.session_id
    assert summaries[0].pipeline_version == "v3.3"


def test_missing_session_raises_not_found(local_tmp_path: Path) -> None:
    manager = SessionManager(Database(local_tmp_path / "hummuse.db"))

    with pytest.raises(SessionNotFoundError):
        manager.get_session("00000000-0000-0000-0000-000000000000")


def test_update_rejects_session_id_mismatch(local_tmp_path: Path) -> None:
    manager = SessionManager(Database(local_tmp_path / "hummuse.db"))
    created = manager.create_session()
    other = manager.create_session()
    wrong_state = _build_state(str(other.session_id))

    with pytest.raises(ValueError):
        manager.update_session(created.session_id, wrong_state)
