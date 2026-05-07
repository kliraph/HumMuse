"""Tests for DQN-backed /chords/from-lyrics wiring."""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path


def _install_audio_dependency_stubs() -> None:
    soundfile = types.ModuleType("soundfile")

    class LibsndfileError(Exception):
        pass

    soundfile.LibsndfileError = LibsndfileError
    soundfile.read = lambda *args, **kwargs: (_empty_audio(), 16000)
    sys.modules.setdefault("soundfile", soundfile)

    noisereduce = types.ModuleType("noisereduce")
    noisereduce.reduce_noise = lambda y, sr, stationary=True, prop_decrease=0.8: y
    sys.modules.setdefault("noisereduce", noisereduce)

    pyloudnorm = types.ModuleType("pyloudnorm")
    pyloudnorm.Meter = lambda sample_rate: types.SimpleNamespace(integrated_loudness=lambda waveform: -16.0)
    pyloudnorm.normalize = types.SimpleNamespace(loudness=lambda waveform, loudness, target_lufs: waveform)
    sys.modules.setdefault("pyloudnorm", pyloudnorm)

    midiutil = types.ModuleType("midiutil")

    class MIDIFile:
        def __init__(self, *args, **kwargs):
            pass

        def addTrackName(self, *args, **kwargs):
            pass

        def addTempo(self, *args, **kwargs):
            pass

        def addNote(self, *args, **kwargs):
            pass

        def writeFile(self, buffer):
            buffer.write(b"MThd")

    midiutil.MIDIFile = MIDIFile
    sys.modules.setdefault("midiutil", midiutil)

    multipart = types.ModuleType("multipart")
    multipart.__version__ = "0.0.0-test"
    multipart_submodule = types.ModuleType("multipart.multipart")
    multipart_submodule.parse_options_header = lambda value: (value, {})
    multipart.multipart = multipart_submodule
    sys.modules.setdefault("multipart", multipart)
    sys.modules.setdefault("multipart.multipart", multipart_submodule)


def _empty_audio():
    import numpy as np

    return np.zeros(1, dtype=np.float32)


_install_audio_dependency_stubs()

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as backend_app  # noqa: E402
from shared.schemas import (  # noqa: E402
    ChordDistribution,
    ChordAnnotation,
    ChordProgression,
    ChordSymbolDerivation,
    NoteEvent,
)


def reset_storage() -> None:
    storage = Path("storage")
    storage.mkdir(parents=True, exist_ok=True)
    backend_app.database._init_db()
    backend_app.experiment_logger.run_store._init_db()
    for db_path in (backend_app.database.db_path, backend_app.experiment_logger.run_store.db_path):
        with sqlite3.connect(db_path) as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")


def _fake_progression(index: int) -> ChordProgression:
    distribution = ChordDistribution(
        position=0,
        rest={"rest": 0.01, "chord": 0.99},
        octave={"2": 0.1, "3": 0.9},
        inversion={"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0},
        pitch_class={
            "C": 0.3,
            "C#": 0.01,
            "D": 0.01,
            "D#": 0.01,
            "E": 0.25,
            "F": 0.01,
            "F#": 0.01,
            "G": 0.25,
            "G#": 0.01,
            "A": 0.01,
            "A#": 0.01,
            "B": 0.01,
        },
        pitch_class_membership=[0.3, 0.01, 0.01, 0.01, 0.25, 0.01, 0.01, 0.25, 0.01, 0.01, 0.01, 0.01, 0.1],
        pitch_class_labels=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "triad_sentinel"],
        selected={"is_rest": False, "pcs": [0, 4, 7], "pc_names": ["C", "E", "G"], "inversion": 0},
    )
    derivation = ChordSymbolDerivation(
        symbol="C",
        root="C",
        root_pc=0,
        quality="major",
        bass="C",
        bass_pc=0,
        pitch_classes=[0, 4, 7],
        inversion=0,
        roman_numeral="I",
        harmonic_function="tonic",
        confidence=0.5,
    )
    annotation = ChordAnnotation(
        position=0,
        symbol="C",
        root="C",
        quality="major",
        bass="C",
        roman_numeral="I",
        function_label="tonic",
        alignment_percentage=1.0,
        strong_beat_notes=[],
        template_phrase="tonic - stable home base",
        derivation=derivation,
    )
    return ChordProgression(
        chords=["C", "G", "Am", "F"],
        score=0.9 - (index * 0.1),
        explanation="Fake DQN progression for endpoint wiring.",
        native_distributions=[distribution],
        chord_annotations=[annotation],
    )


def test_chords_from_lyrics_uses_dqn_when_session_has_melody(monkeypatch) -> None:
    reset_storage()
    client = TestClient(backend_app.app)
    monkeypatch.setattr(
        backend_app,
        "generate_dqn_chords",
        lambda melody_notes, key, emotion_vector, top_k, tempo_bpm: [
            _fake_progression(index)
            for index in range(top_k)
        ],
    )

    created = client.post("/session/create", json={"lyrics_text": "Sunrise love"})
    assert created.status_code == 200
    session_id = created.json()["state"]["session_id"]

    state = backend_app.session_manager.get_session(session_id)
    state.melody_notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95)
    ]
    state.detected_key = "C major"
    state.detected_tempo = 120.0
    backend_app.session_manager.update_session(session_id, state)

    response = client.post(
        "/chords/from-lyrics",
        json={"session_id": session_id, "text": "Love under the sunrise"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["chord_progressions"]) == 3
    assert body["top_progressions"][0] == ["C", "G", "Am", "F"]
    first_progression = body["chord_progressions"][0]
    assert first_progression["native_distributions"]
    assert first_progression["chord_annotations"][0]["symbol"] == "C"

    saved_state = client.get(f"/session/{session_id}/state").json()["state"]
    saved_progression = saved_state["chord_progressions"][0]
    assert saved_progression["native_distributions"]
    assert saved_progression["chord_annotations"]
    chord_theory = saved_state["explanation_report"]["chord_theory"][0]
    assert chord_theory["native_distribution_count"] == 1
    assert chord_theory["annotations"][0]["symbol"] == "C"




