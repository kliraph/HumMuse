"""Tests for continuation candidate constraint filtering."""

from __future__ import annotations

import json
from pathlib import Path

from symusic import Note, Score, Track

from ml.melody_sketchpad.continuation.constraints import filter_candidates
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import ChordProgression, NoteEvent


def test_ensemble_candidate_pool_filters_to_three_to_five_survivors_with_reasons(tmp_path: Path) -> None:
    primer_notes = _primer_notes()
    candidates = [
        _candidate(tmp_path, "A", 0, [60, 62, 64, 67]),  # pass
        _candidate(tmp_path, "B", 0, [64, 65, 67, 64]),  # pass
        _candidate(tmp_path, "C", 0, [67, 65, 64, 60]),  # pass
        _candidate(tmp_path, "A", 1, [61, 63, 66, 70]),  # key/range fail
        _candidate(tmp_path, "B", 1, [72, 71, 72, 71]),  # range fail
        _candidate(tmp_path, "C", 1, [62, 64, 65, 67]),  # consonance fail
        _candidate(tmp_path, "A", 2, [60, 62, 64, 67], duration=0.5, step=0.5),  # density fail
        _candidate(tmp_path, "B", 2, [60, 64, 67, 64]),  # pass
    ]

    result = filter_candidates(
        candidates,
        primer_notes=primer_notes,
        primer_profile=build_melody_profile(primer_notes),
        detected_key="C major",
        chord_progressions=[_progression()],
    )

    assert 3 <= len(result.survivors) <= 5
    assert len(result.survivors) == 4
    assert all(candidate["constraint_explanation"] for candidate in result.survivors)
    rejected = [entry for entry in result.trace if entry["status"] == "rejected"]
    assert len(rejected) == 4
    assert all(entry["reasons"] for entry in rejected)
    assert len(result.rejection_metadata) == len(rejected)
    assert result.rejection_metadata == [
        {
            "model_id": entry["model_id"],
            "temperature": 0.8,
            "rejection_reason": "; ".join(entry["reasons"]),
            "candidate_idx": entry["candidate"],
        }
        for entry in rejected
    ]
    json.dumps(result.rejection_metadata)


def test_single_model_fallback_filters_to_two_to_three_survivors(tmp_path: Path) -> None:
    primer_notes = _primer_notes()
    candidates = [
        _candidate(tmp_path, "single", 0, [60, 62, 64, 67]),  # pass
        _candidate(tmp_path, "single", 1, [64, 65, 67, 64]),  # pass
        _candidate(tmp_path, "single", 2, [67, 65, 64, 60]),  # pass
        _candidate(tmp_path, "single", 3, [61, 63, 66, 70]),
        _candidate(tmp_path, "single", 4, [72, 71, 72, 71]),
        _candidate(tmp_path, "single", 5, [62, 64, 65, 67]),
        _candidate(tmp_path, "single", 6, [60, 62, 64, 67], duration=0.5, step=0.5),
        _candidate(tmp_path, "single", 7, [66, 65, 66, 65]),
    ]

    result = filter_candidates(
        candidates,
        primer_notes=primer_notes,
        primer_profile=build_melody_profile(primer_notes),
        detected_key="C major",
        chord_progressions=[_progression()],
        max_survivors=3,
    )

    assert 2 <= len(result.survivors) <= 3
    assert len(result.survivors) == 3
    assert len(result.trace) == 8
    assert len(result.rejection_metadata) == 5
    assert all(set(item) == {"model_id", "temperature", "rejection_reason", "candidate_idx"} for item in result.rejection_metadata)


def _primer_notes() -> list[NoteEvent]:
    return [
        NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=62, onset=1.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=64, onset=2.0, duration=1.0, velocity=96, confidence=1.0),
        NoteEvent(pitch=64, onset=3.0, duration=1.0, velocity=96, confidence=1.0),
    ]


def _candidate(
    tmp_path: Path,
    model_id: str,
    index: int,
    pitches: list[int],
    *,
    duration: float = 1.0,
    step: float = 1.0,
) -> dict:
    midi_path = tmp_path / f"{model_id}_{index}.mid"
    _write_midi(midi_path, pitches, duration=duration, step=step)
    return {
        "midi_path": str(midi_path),
        "model_id": model_id,
        "temperature": 0.8,
        "tokens": [index],
        "log_prob": -0.1 * index,
    }


def _write_midi(path: Path, pitches: list[int], *, duration: float, step: float) -> None:
    score = Score(480)
    track = Track(name="candidate", program=0, is_drum=False)
    for note_index, pitch in enumerate(pitches):
        track.notes.append(
            Note(
                int(round(note_index * step * 480)),
                int(round(duration * 480)),
                pitch,
                96,
            )
        )
    score.tracks.append(track)
    score.dump_midi(str(path))


def _progression() -> ChordProgression:
    return ChordProgression(
        chords=["C", "F", "G", "C"],
        score=0.9,
        explanation="Fixture progression.",
    )
