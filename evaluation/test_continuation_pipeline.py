"""Tests for end-to-end continuation pipeline assembly."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from symusic import Note, Score, Track

import ml.melody_sketchpad.continuation.pipeline as pipeline_module
from ml.melody_sketchpad.continuation.inference import (
    EnsembleMember,
    EnsembleMelodyContinuationModel,
    MelodyContinuationModel,
)
from ml.melody_sketchpad.continuation.pipeline import (
    build_continuation_model_from_config,
    continue_melody,
    load_continuation_config,
)
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import ChordProgression, EmotionVector, NoteEvent, SessionState


def test_continue_melody_returns_top_three_music_transformer_suggestions(tmp_path: Path) -> None:
    state = _session_state()
    generator = _fake_ensemble()

    started = time.perf_counter()
    suggestions = continue_melody(state, generator=generator, out_dir=tmp_path, max_new_tokens=16)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert len(suggestions) == 3
    assert all(suggestion.engine == "music_transformer" for suggestion in suggestions)
    assert all(suggestion.model_id in {"A", "B", "C"} for suggestion in suggestions)
    assert all(suggestion.avg_log_prob is not None for suggestion in suggestions)
    assert all(suggestion.constraint_trace for suggestion in suggestions)
    assert all(suggestion.score_breakdown for suggestion in suggestions)
    assert all(suggestion.midi_bytes.startswith(b"MThd") for suggestion in suggestions)
    assert state.user_params["continuation_candidate_count"] == 8
    assert 3 <= state.user_params["continuation_survivor_count"] <= 5
    assert state.user_params["continuation_rejection_metadata"]


def test_load_continuation_config_reads_engine_config(tmp_path: Path) -> None:
    config_path = tmp_path / "continuation.json"
    config_path.write_text('{"continuation": {"engine": "ensemble"}}', encoding="utf-8")

    config = load_continuation_config(config_path)

    assert config["continuation"]["engine"] == "ensemble"
    assert config["continuation"]["single"]["model_id"] == "single"
    assert len(config["continuation"]["ensemble"]["members"]) == 3


def test_configured_single_model_builds_one_member_ensemble(monkeypatch) -> None:
    class CapturedEnsemble:
        def __init__(self, members, device, top_k) -> None:
            self.members = list(members)
            self.device = device
            self.top_k = top_k

    monkeypatch.setattr(pipeline_module, "EnsembleMelodyContinuationModel", CapturedEnsemble)

    model = build_continuation_model_from_config(
        {
            "continuation": {
                "engine": "single",
                "device": "cpu",
                "top_k": 17,
                "single": {
                    "checkpoint": "storage/models/Lakh-MT/best_seed1337.pt",
                    "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
                    "model_id": "solo",
                    "count": 6,
                    "temperatures": [0.6, 0.8],
                },
            }
        }
    )

    assert len(model.members) == 1
    member = model.members[0]
    assert member.checkpoint_path == pipeline_module._REPO_ROOT / "storage/models/Lakh-MT/best_seed1337.pt"
    assert member.tokenizer_path == pipeline_module._REPO_ROOT / "storage/models/Lakh-MT/tokenizer.json"
    assert member.model_id == "solo"
    assert member.count == 6
    assert member.temperatures == (0.6, 0.8)
    assert model.top_k == 17


def test_configured_ensemble_model_uses_member_spec(monkeypatch) -> None:
    class CapturedEnsemble:
        def __init__(self, members, device, top_k) -> None:
            self.members = list(members)
            self.device = device
            self.top_k = top_k

    monkeypatch.setattr(pipeline_module, "EnsembleMelodyContinuationModel", CapturedEnsemble)

    model = build_continuation_model_from_config(
        {
            "continuation": {
                "engine": "ensemble",
                "device": "cpu",
                "top_k": 23,
                "ensemble": {
                    "members": [
                        {
                            "checkpoint": "storage/models/Lakh-MT/best_seed7.pt",
                            "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
                            "model_id": "A",
                            "temperatures": [0.5, 0.7, 0.9],
                            "count": 3,
                        },
                        {
                            "checkpoint": "storage/models/Lakh-MT/best_seed42.pt",
                            "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
                            "model_id": "B",
                            "temperatures": [0.7, 0.9],
                            "count": 2,
                        },
                    ]
                },
            }
        }
    )

    assert model.top_k == 23
    assert [member.model_id for member in model.members] == ["A", "B"]
    assert [member.count for member in model.members] == [3, 2]
    assert [member.temperatures for member in model.members] == [(0.5, 0.7, 0.9), (0.7, 0.9)]
    assert model.members[0].checkpoint_path == pipeline_module._REPO_ROOT / "storage/models/Lakh-MT/best_seed7.pt"


class _FakeMemberModel:
    _PATTERNS = {
        "A": [[60, 62, 64, 67], [61, 63, 66, 70], [67, 65, 64, 60]],
        "B": [[64, 65, 67, 64], [72, 71, 72, 71], [62, 64, 65, 67]],
        "C": [[60, 64, 67, 64], [60, 62, 64, 67]],
    }

    def generate_candidates(
        self,
        prompt_midi_path,
        n_candidates: int,
        max_new_tokens: int,
        temperatures,
        out_dir,
        model_id: str = "single",
    ) -> list[dict]:
        out_dir = Path(out_dir)
        candidates = []
        idx = 0
        patterns = self._PATTERNS[model_id]
        for temp, count in zip(temperatures, MelodyContinuationModel._allocate(n_candidates, len(temperatures))):
            for _ in range(count):
                midi_path = out_dir / f"{model_id}_{idx:02d}_t{temp:.2f}.mid"
                _write_midi(midi_path, patterns[idx % len(patterns)])
                candidates.append(
                    {
                        "midi_path": str(midi_path),
                        "tokens": [idx],
                        "temperature": float(temp),
                        "log_prob": -0.1 * idx,
                        "model_id": model_id,
                    }
                )
                idx += 1
        return candidates


def _fake_ensemble() -> EnsembleMelodyContinuationModel:
    members = [
        EnsembleMember("a.pt", "tok.json", "A", (0.5, 0.7, 0.9), 3),
        EnsembleMember("b.pt", "tok.json", "B", (0.5, 0.7, 0.9), 3),
        EnsembleMember("c.pt", "tok.json", "C", (0.7, 0.9), 2),
    ]
    ensemble = object.__new__(EnsembleMelodyContinuationModel)
    ensemble.members = members
    ensemble.models = {member.model_id: _FakeMemberModel() for member in members}
    return ensemble


def _session_state() -> SessionState:
    notes = [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=62, onset=1.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=64, onset=2.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=64, onset=3.0, duration=1.0, velocity=96, confidence=1.0),
    ]
    return SessionState(
        session_id=uuid4(),
        melody_notes=notes,
        melody_profile=build_melody_profile(notes),
        detected_key="C major",
        detected_tempo=120.0,
        chord_progressions=[
            ChordProgression(chords=["C", "F", "G", "C"], score=0.9, explanation="Fixture chords.")
        ],
        emotion_vector=EmotionVector(valence=0.2, arousal=0.4),
    )


def _write_midi(path: Path, pitches: list[int]) -> None:
    score = Score(480)
    track = Track(name="candidate", program=0, is_drum=False)
    for index, pitch in enumerate(pitches):
        track.notes.append(Note(index * 480, 480, pitch, 96))
    score.tracks.append(track)
    score.dump_midi(str(path))
