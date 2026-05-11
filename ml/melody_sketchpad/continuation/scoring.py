"""Profile-based scoring for melody-continuation candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import MelodyProfile, NoteEvent

_EPSILON = 1e-12


@dataclass(frozen=True)
class ContinuationScore:
    """Serializable score breakdown for one continuation candidate."""

    score: float
    js_divergence: float
    js_similarity: float
    contour_variety: float
    range_usage: float
    candidate_profile: MelodyProfile

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_profile"] = self.candidate_profile.model_dump()
        return data


def score_candidate(
    candidate_notes: list[NoteEvent],
    primer_profile: MelodyProfile,
) -> ContinuationScore:
    """
    Score a decoded continuation against the primer melody profile.

    The score combines interval-distribution similarity via Jensen-Shannon
    divergence, contour variety, and pitch-range usage. It is representation
    agnostic: callers pass decoded `NoteEvent` lists, not tokenizer ids.
    """
    candidate_profile = build_melody_profile(candidate_notes)
    js_divergence = jensen_shannon_divergence(
        primer_profile.interval_histogram,
        candidate_profile.interval_histogram,
    )
    js_similarity = 1.0 - js_divergence
    contour_variety = _contour_variety(primer_profile.contour, candidate_profile.contour)
    range_usage = _range_usage(primer_profile.pitch_range, candidate_profile.pitch_range)
    score = (
        0.60 * js_similarity
        + 0.20 * contour_variety
        + 0.20 * range_usage
    )
    return ContinuationScore(
        score=round(_clamp01(score), 6),
        js_divergence=round(_clamp01(js_divergence), 6),
        js_similarity=round(_clamp01(js_similarity), 6),
        contour_variety=round(_clamp01(contour_variety), 6),
        range_usage=round(_clamp01(range_usage), 6),
        candidate_profile=candidate_profile,
    )


def score_candidates(
    candidates: list[dict[str, Any]],
    primer_profile: MelodyProfile,
) -> list[dict[str, Any]]:
    """Attach score breakdowns to candidate dicts and return best-first order."""
    scored: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        notes = candidate.get("notes", [])
        breakdown = score_candidate(notes, primer_profile)
        enriched = dict(candidate)
        enriched["profile_score"] = breakdown.score
        enriched["score_breakdown"] = breakdown.as_dict()
        enriched["_score_input_index"] = index
        scored.append(enriched)

    ranked = sorted(
        scored,
        key=lambda item: (
            float(item["profile_score"]),
            float(item.get("constraint_score", 0.0)),
            float(item.get("log_prob", 0.0)),
            -int(item["_score_input_index"]),
        ),
        reverse=True,
    )
    for item in ranked:
        item.pop("_score_input_index", None)
    return ranked


def jensen_shannon_divergence(p: list[float], q: list[float]) -> float:
    """
    Jensen-Shannon divergence over two discrete distributions, base-2 normalized.

    Returns a value in [0, 1], where 0 means identical distributions.
    """
    left, right = _aligned_distributions(p, q)
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    return _clamp01(0.5 * _kl_divergence(left, midpoint) + 0.5 * _kl_divergence(right, midpoint))


def _aligned_distributions(p: list[float], q: list[float]) -> tuple[list[float], list[float]]:
    size = max(len(p), len(q), 1)
    left = [max(0.0, float(p[index])) if index < len(p) else 0.0 for index in range(size)]
    right = [max(0.0, float(q[index])) if index < len(q) else 0.0 for index in range(size)]
    return _normalize(left), _normalize(right)


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in values]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    total = 0.0
    for left, right in zip(p, q):
        if left <= 0:
            continue
        total += left * math.log(left / max(right, _EPSILON), 2)
    return total


def _contour_variety(primer_contour: str, candidate_contour: str) -> float:
    if primer_contour == candidate_contour:
        return 0.75
    if {primer_contour, candidate_contour} in ({"rising", "falling"}, {"arch", "valley"}):
        return 0.85
    return 1.0


def _range_usage(primer_range: tuple[int, int], candidate_range: tuple[int, int]) -> float:
    primer_span = max(0, int(primer_range[1]) - int(primer_range[0]))
    candidate_span = max(0, int(candidate_range[1]) - int(candidate_range[0]))
    denominator = max(primer_span, candidate_span, 1)
    return _clamp01(1.0 - abs(candidate_span - primer_span) / denominator)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
