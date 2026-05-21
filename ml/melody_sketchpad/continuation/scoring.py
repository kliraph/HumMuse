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
    # Per-axis proximity to (primer + transition delta) target, in [0, 1].
    # Populated only when scoring is section-aware (transition + primer_notes
    # both provided). None means primer-relative scoring was used.
    section_alignment: float | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_profile"] = self.candidate_profile.model_dump()
        return data


# Blend weights when transition is available. 0.5/0.5 makes the section
# bias actually flip rankings — a 0.6/0.4 weighting still leaves the
# primer-relative term dominant, so a verse-twin candidate (JS_similarity=1.0
# against the primer) outranks a chorus-shaped one. Splitting evenly lets
# alignment-along-the-delta-axes drive ranking when sections are picked,
# without throwing away the primer-similarity signal entirely.
_PRIMER_RELATIVE_WEIGHT = 0.50
_SECTION_ALIGNMENT_WEIGHT = 0.50
# z-score cap shared with the conditional constraint checks; keeps the two
# stages on the same scale so "passes constraints + ranks high on alignment"
# stays a meaningful claim.
_ALIGNMENT_Z_MAX = 2.0


def score_candidate(
    candidate_notes: list[NoteEvent],
    primer_profile: MelodyProfile,
    *,
    primer_notes: list[NoteEvent] | None = None,
    transition: dict[str, Any] | None = None,
) -> ContinuationScore:
    """
    Score a decoded continuation against the primer melody profile.

    The primer-relative score combines interval-distribution similarity via
    Jensen-Shannon divergence, contour variety, and pitch-range usage. It is
    representation agnostic: callers pass decoded `NoteEvent` lists.

    When `transition` and `primer_notes` are both provided, the final score
    blends the primer-relative score with a section-alignment score that
    rewards candidates close to (primer + delta) along the four BiMMuDa
    transition axes. This is what biases ranking toward the chosen target
    section (e.g. verse->chorus prefers lifted, slightly-tighter candidates).
    """
    candidate_profile = build_melody_profile(candidate_notes)
    js_divergence = jensen_shannon_divergence(
        primer_profile.interval_histogram,
        candidate_profile.interval_histogram,
    )
    js_similarity = 1.0 - js_divergence
    contour_variety = _contour_variety(primer_profile.contour, candidate_profile.contour)
    range_usage = _range_usage(primer_profile.pitch_range, candidate_profile.pitch_range)
    primer_relative = (
        0.60 * js_similarity
        + 0.20 * contour_variety
        + 0.20 * range_usage
    )

    section_alignment: float | None
    if transition is not None and primer_notes:
        section_alignment = _section_alignment_score(
            candidate_notes=candidate_notes,
            primer_notes=primer_notes,
            candidate_profile=candidate_profile,
            primer_profile=primer_profile,
            transition=transition,
        )
        final = (
            _PRIMER_RELATIVE_WEIGHT * primer_relative
            + _SECTION_ALIGNMENT_WEIGHT * section_alignment
        )
    else:
        section_alignment = None
        final = primer_relative

    return ContinuationScore(
        score=round(_clamp01(final), 6),
        js_divergence=round(_clamp01(js_divergence), 6),
        js_similarity=round(_clamp01(js_similarity), 6),
        contour_variety=round(_clamp01(contour_variety), 6),
        range_usage=round(_clamp01(range_usage), 6),
        candidate_profile=candidate_profile,
        section_alignment=None if section_alignment is None else round(_clamp01(section_alignment), 6),
    )


def score_candidates(
    candidates: list[dict[str, Any]],
    primer_profile: MelodyProfile,
    *,
    primer_notes: list[NoteEvent] | None = None,
    transition: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach score breakdowns to candidate dicts and return best-first order."""
    scored: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        notes = candidate.get("notes", [])
        breakdown = score_candidate(
            notes,
            primer_profile,
            primer_notes=primer_notes,
            transition=transition,
        )
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


def _section_alignment_score(
    *,
    candidate_notes: list[NoteEvent],
    primer_notes: list[NoteEvent],
    candidate_profile: MelodyProfile,
    primer_profile: MelodyProfile,
    transition: dict[str, Any],
) -> float:
    """
    Mean per-axis proximity to (primer + transition delta) target, in [0, 1].

    One score per BiMMuDa delta axis (register, pitch span, density, boundary
    interval). Each is `max(0, 1 - z / Z_CAP)`, i.e. the same penalty shape
    used by the conditional constraint checks. Averaged uniformly across the
    four axes — no per-axis weighting (tuning that would need a separate
    calibration study).
    """
    z_cap = _ALIGNMENT_Z_MAX

    # Register: mean pitch of the section.
    primer_register = sum(int(note.pitch) for note in primer_notes) / max(len(primer_notes), 1)
    candidate_register = (
        sum(int(note.pitch) for note in candidate_notes) / len(candidate_notes)
        if candidate_notes
        else primer_register
    )
    register_delta = transition["delta_register_semitones"]
    register_score = _axis_alignment(
        observed=candidate_register,
        expected=primer_register + float(register_delta["mean"]),
        std=float(register_delta["std"]),
        z_cap=z_cap,
    )

    # Pitch span: max - min pitch.
    primer_pitches = [int(note.pitch) for note in primer_notes]
    candidate_pitches = [int(note.pitch) for note in candidate_notes]
    primer_span = (max(primer_pitches) - min(primer_pitches)) if primer_pitches else 0
    candidate_span = (max(candidate_pitches) - min(candidate_pitches)) if candidate_pitches else primer_span
    span_delta = transition["delta_pitch_range_semitones"]
    span_score = _axis_alignment(
        observed=candidate_span,
        expected=primer_span + float(span_delta["mean"]),
        std=float(span_delta["std"]),
        z_cap=z_cap,
    )

    # Rhythmic density.
    density_delta = transition["delta_density_notes_per_bar"]
    density_score = _axis_alignment(
        observed=float(candidate_profile.rhythmic_density),
        expected=float(primer_profile.rhythmic_density) + float(density_delta["mean"]),
        std=float(density_delta["std"]),
        z_cap=z_cap,
    )

    # Boundary interval (absolute magnitude — matches upstream prior script).
    if primer_notes and candidate_notes:
        primer_last = max(primer_notes, key=lambda note: note.onset + note.duration).pitch
        candidate_first = min(candidate_notes, key=lambda note: note.onset).pitch
        boundary_observed = abs(int(candidate_first) - int(primer_last))
        boundary_profile = transition["boundary_interval_semitones"]
        boundary_score = _axis_alignment(
            observed=float(boundary_observed),
            expected=float(boundary_profile["mean"]),
            std=float(boundary_profile["std"]),
            z_cap=z_cap,
        )
    else:
        boundary_score = 1.0

    return (register_score + span_score + density_score + boundary_score) / 4.0


def _axis_alignment(*, observed: float, expected: float, std: float, z_cap: float) -> float:
    z = abs(observed - expected) / max(std, 1e-6)
    return _clamp01(1.0 - z / z_cap)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
