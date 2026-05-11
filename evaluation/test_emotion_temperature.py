"""Tests for emotion-conditioned continuation temperature schedules."""

from __future__ import annotations

from statistics import pvariance
from typing import Sequence

from ml.melody_sketchpad.continuation.emotion_temperature import (
    emotion_temperature_schedule,
    generate_ensemble_with_emotion_temperature,
    rescale_temperature_schedule,
)
from ml.melody_sketchpad.continuation.inference import (
    EnsembleMember,
    EnsembleMelodyContinuationModel,
    MelodyContinuationModel,
)
from shared.schemas import EmotionVector


def test_high_arousal_widens_temperature_schedule() -> None:
    low = emotion_temperature_schedule(EmotionVector(valence=0.0, arousal=0.1))
    high = emotion_temperature_schedule(EmotionVector(valence=0.0, arousal=0.9))

    assert high[0] < low[0]
    assert high[-1] > low[-1]
    assert _width(high) > _width(low)


def test_high_arousal_yields_broader_sampled_pitch_variance() -> None:
    """
    The ensemble path is the only continuation entry point, so this test
    exercises a one-member ensemble — the same shape the pipeline builds
    for engine="single".
    """
    low_ensemble = _single_member_ensemble()
    high_ensemble = _single_member_ensemble()

    low = generate_ensemble_with_emotion_temperature(
        low_ensemble,
        "primer.mid",
        emotion_vector=EmotionVector(valence=0.0, arousal=0.1),
    )
    high = generate_ensemble_with_emotion_temperature(
        high_ensemble,
        "primer.mid",
        emotion_vector=EmotionVector(valence=0.0, arousal=0.9),
    )

    assert pvariance(_first_pitches(high)) > pvariance(_first_pitches(low))


def test_high_arousal_widens_each_member_schedule_without_changing_counts_or_order() -> None:
    members = [
        EnsembleMember("a.pt", "tok.json", "A", (0.5, 0.7, 0.9), 3),
        EnsembleMember("b.pt", "tok.json", "B", (0.6, 0.8, 1.0), 3),
        EnsembleMember("c.pt", "tok.json", "C", (0.7, 0.9), 2),
    ]
    ensemble = object.__new__(EnsembleMelodyContinuationModel)
    ensemble.members = members
    ensemble.models = {member.model_id: _TemperaturePitchModel() for member in members}

    generated = generate_ensemble_with_emotion_temperature(
        ensemble,
        "primer.mid",
        emotion_vector=EmotionVector(valence=0.0, arousal=0.9),
        max_new_tokens=32,
        out_dir="out",
    )

    assert [candidate["model_id"] for candidate in generated] == ["A", "B", "C", "A", "B", "C", "A", "B"]
    assert [candidate["temperature"] for candidate in generated if candidate["model_id"] == "A"] == [0.4, 0.7, 1.0]
    assert [candidate["temperature"] for candidate in generated if candidate["model_id"] == "B"] == [0.5, 0.8, 1.1]
    assert [candidate["temperature"] for candidate in generated if candidate["model_id"] == "C"] == [0.65, 0.95]
    assert [member.count for member in ensemble.members] == [3, 3, 2]
    assert [member.temperatures for member in ensemble.members] == [(0.5, 0.7, 0.9), (0.6, 0.8, 1.0), (0.7, 0.9)]


def test_rescale_temperature_schedule_preserves_member_center() -> None:
    base = (0.5, 0.7, 0.9)
    high = rescale_temperature_schedule(base, EmotionVector(valence=0.0, arousal=0.9))

    assert high == (0.4, 0.7, 1.0)
    assert sum(high) / len(high) == sum(base) / len(base)


class _TemperaturePitchModel:
    def __init__(self) -> None:
        self.last_call = {}

    def generate_candidates(
        self,
        prompt_midi_path,
        n_candidates: int = 8,
        max_new_tokens: int = 128,
        temperatures: Sequence[float] = (0.8, 0.9, 1.0),
        out_dir=None,
        model_id: str = "single",
    ) -> list[dict]:
        self.last_call = {
            "prompt_midi_path": prompt_midi_path,
            "n_candidates": n_candidates,
            "max_new_tokens": max_new_tokens,
            "temperatures": tuple(temperatures),
            "out_dir": out_dir,
            "model_id": model_id,
        }
        candidates = []
        idx = 0
        for temp, count in zip(
            temperatures,
            MelodyContinuationModel._allocate(n_candidates, len(temperatures)),
        ):
            for _ in range(count):
                pitch = int(round(60 + ((float(temp) - 0.85) * 24)))
                candidates.append({"notes": [{"pitch": pitch}], "temperature": float(temp), "model_id": model_id})
                idx += 1
        return candidates


def _single_member_ensemble() -> EnsembleMelodyContinuationModel:
    member = EnsembleMember("a.pt", "tok.json", "single", (0.5, 0.7, 0.9), 12)
    ensemble = object.__new__(EnsembleMelodyContinuationModel)
    ensemble.members = [member]
    ensemble.models = {"single": _TemperaturePitchModel()}
    return ensemble


def _first_pitches(candidates: list[dict]) -> list[int]:
    return [int(candidate["notes"][0]["pitch"]) for candidate in candidates]


def _width(schedule: tuple[float, ...]) -> float:
    return max(schedule) - min(schedule)
