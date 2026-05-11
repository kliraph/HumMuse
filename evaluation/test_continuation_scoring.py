"""Tests for continuation profile scoring."""

from __future__ import annotations

from ml.melody_sketchpad.continuation.scoring import (
    jensen_shannon_divergence,
    score_candidate,
    score_candidates,
)
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import NoteEvent


def test_monotonic_profile_match_has_low_js_divergence_and_high_score() -> None:
    primer = _notes([60, 62, 65, 67])
    candidate = _notes([64, 66, 69, 71])

    score = score_candidate(candidate, build_melody_profile(primer))

    assert score.js_divergence < 0.05
    assert score.score > 0.7
    assert score.candidate_profile.contour == "rising"


def test_profile_matching_candidate_scores_above_threshold() -> None:
    primer = _notes([60, 67, 64, 62])
    matching_candidate = _notes([62, 69, 66, 64])
    unrelated_candidate = _notes([72, 60, 72, 60])

    primer_profile = build_melody_profile(primer)
    matching_score = score_candidate(matching_candidate, primer_profile)
    unrelated_score = score_candidate(unrelated_candidate, primer_profile)

    assert matching_score.score > 0.7
    assert matching_score.score > unrelated_score.score


def test_score_candidates_attaches_breakdown_and_ranks_best_first() -> None:
    primer_profile = build_melody_profile(_notes([60, 62, 65, 67]))
    candidates = [
        {"notes": _notes([72, 60, 72, 60]), "model_id": "wide"},
        {"notes": _notes([64, 66, 69, 71]), "model_id": "match"},
    ]

    ranked = score_candidates(candidates, primer_profile)

    assert ranked[0]["model_id"] == "match"
    assert ranked[0]["profile_score"] > 0.7
    assert "js_divergence" in ranked[0]["score_breakdown"]


def test_jensen_shannon_divergence_is_zero_for_identical_distributions() -> None:
    assert jensen_shannon_divergence([0.0, 0.5, 0.5], [0.0, 0.5, 0.5]) == 0.0


def _notes(pitches: list[int]) -> list[NoteEvent]:
    return [
        NoteEvent(
            pitch=pitch,
            onset=float(index),
            duration=1.0,
            velocity=96,
            confidence=1.0,
        )
        for index, pitch in enumerate(pitches)
    ]
