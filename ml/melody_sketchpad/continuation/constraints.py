"""Constraint filtering for decoded melody-continuation candidates."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from symusic import Score

from ml.harmony.chord_symbols import NAME_TO_PC, QUALITY_TEMPLATES, parse_key
from ml.melody_sketchpad.continuation.priors import load_transition_profile
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import ChordProgression, MelodyProfile, NoteEvent

_DEFAULT_MAX_SURVIVORS = 5
_KEY_ADHERENCE_MIN_RATIO = 0.75
_PITCH_RANGE_SEMITONES = 4
_DENSITY_TOLERANCE = 0.20
_STRONG_BEAT_TOLERANCE = 0.08
_BEATS_PER_BAR = 4.0
# z-score cap for conditional-prior checks. Wide on purpose: BiMMuDa stds
# are large relative to the means, so a tight z (e.g. 1.5) would reject
# most stylistically reasonable continuations.
_CONDITIONAL_Z_MAX = 2.0


@dataclass(frozen=True)
class ConstraintFilterResult:
    """Accepted candidates plus a serializable per-candidate constraint trace."""

    survivors: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    rejection_metadata: list[dict[str, Any]]


def filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    primer_notes: list[NoteEvent],
    primer_profile: MelodyProfile,
    detected_key: str | None,
    chord_progressions: list[ChordProgression] | None = None,
    max_survivors: int = _DEFAULT_MAX_SURVIVORS,
    primer_section: str | None = None,
    target_section: str | None = None,
) -> ConstraintFilterResult:
    """
    Decode and filter continuation candidates using session-level constraints.

    When `primer_section` and `target_section` are both set AND the
    BiMMuDa-derived transition prior for that pair meets the minimum sample
    threshold, the density / pitch-range primer-relative checks are replaced
    by conditional-on-transition checks, and two new section-aware checks
    fire (register shift, boundary interval). Otherwise the function uses
    purely primer-relative checks (continue stylistically near the primer).

    Rejected candidates are retained in `trace` with reason strings and in
    `rejection_metadata` as compact run-log records. Accepted candidates are
    returned with decoded `notes`, `constraint_score`, `constraint_trace`, and
    an explanation-friendly `constraint_explanation`.
    """
    if max_survivors <= 0:
        raise ValueError("max_survivors must be positive")

    primer_last_pitch = _last_pitch(primer_notes)
    chord_symbols = _first_progression_symbols(chord_progressions or [])
    transition = load_transition_profile(primer_section, target_section)

    accepted: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    rejection_metadata: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("model_id", "single")
        temperature = candidate.get("temperature")
        midi_path = candidate.get("midi_path")
        try:
            notes = decode_candidate_midi(Path(midi_path))
            candidate_profile = build_melody_profile(notes)
            checks = _evaluate_candidate(
                notes=notes,
                candidate_profile=candidate_profile,
                primer_notes=primer_notes,
                primer_profile=primer_profile,
                primer_last_pitch=primer_last_pitch,
                detected_key=detected_key,
                chord_symbols=chord_symbols,
                transition=transition,
            )
            reasons = [check["reason"] for check in checks if not check["passed"]]
            status = "accepted" if not reasons else "rejected"
        except Exception as exc:
            notes = []
            candidate_profile = None
            checks = []
            reasons = [f"decode_failed: {exc}"]
            status = "rejected"

        score = _constraint_score(checks)
        entry = {
            "candidate": index,
            "model_id": candidate_id,
            "midi_path": str(midi_path),
            "status": status,
            "score": round(score, 6),
            "reasons": reasons,
            "checks": checks,
        }
        trace.append(entry)

        if status == "accepted":
            enriched = dict(candidate)
            enriched["notes"] = notes
            enriched["melody_profile"] = candidate_profile
            enriched["constraint_score"] = round(score, 6)
            enriched["constraint_trace"] = entry
            enriched["constraint_explanation"] = format_constraint_trace(entry)
            accepted.append(enriched)
        else:
            rejection_metadata.append(
                _rejection_metadata(
                    candidate_idx=index,
                    model_id=candidate_id,
                    temperature=temperature,
                    reasons=reasons,
                )
            )

    ranked = sorted(
        accepted,
        key=lambda item: (
            item["constraint_score"],
            float(item.get("log_prob", 0.0)),
            -int(item["constraint_trace"]["candidate"]),
        ),
        reverse=True,
    )
    return ConstraintFilterResult(
        survivors=ranked[:max_survivors],
        trace=trace,
        rejection_metadata=rejection_metadata,
    )


def decode_candidate_midi(midi_path: Path | str) -> list[NoteEvent]:
    """Decode a single-track or multi-track MIDI candidate to ordered NoteEvents."""
    score = Score(Path(midi_path))
    tpq = float(getattr(score, "ticks_per_quarter", getattr(score, "tpq", 480)) or 480)
    notes: list[NoteEvent] = []
    for track in score.tracks:
        if getattr(track, "is_drum", False):
            continue
        for note in track.notes:
            duration = max(float(note.duration) / tpq, 1e-3)
            notes.append(
                NoteEvent(
                    pitch=int(note.pitch),
                    onset=float(note.start) / tpq,
                    duration=duration,
                    velocity=max(1, min(127, int(note.velocity))),
                    confidence=1.0,
                )
            )
    return sorted(notes, key=lambda item: (item.onset, item.pitch, item.duration))


def format_constraint_trace(trace_entry: dict[str, Any]) -> str:
    """Render one trace entry for MelodySuggestion.explanation."""
    if trace_entry["status"] == "accepted":
        return f"Constraint filter accepted candidate {trace_entry['candidate']} with score {trace_entry['score']:.3f}."
    reasons = "; ".join(trace_entry.get("reasons", [])) or "unknown reason"
    return f"Constraint filter rejected candidate {trace_entry['candidate']}: {reasons}."


def _evaluate_candidate(
    *,
    notes: list[NoteEvent],
    candidate_profile: MelodyProfile,
    primer_notes: list[NoteEvent],
    primer_profile: MelodyProfile,
    primer_last_pitch: int | None,
    detected_key: str | None,
    chord_symbols: list[str],
    transition: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not notes:
        return [_check("nonempty_midi", False, 0.0, "candidate MIDI contains no melody notes")]

    if transition is None:
        # Primer-relative branch: continue stylistically near the primer.
        return [
            _key_adherence_check(notes, detected_key),
            _pitch_range_check(notes, primer_last_pitch),
            _density_check(candidate_profile, primer_profile),
            _chord_consonance_check(notes, chord_symbols),
        ]

    # Section-conditional branch: apply BiMMuDa deltas on top of the primer.
    return [
        _key_adherence_check(notes, detected_key),
        _chord_consonance_check(notes, chord_symbols),
        _conditional_density_check(candidate_profile, primer_profile, transition),
        _conditional_pitch_span_check(notes, primer_notes, transition),
        _register_shift_check(notes, primer_notes, transition),
        _boundary_interval_check(notes, primer_notes, transition),
    ]


def _conditional_density_check(
    candidate_profile: MelodyProfile,
    primer_profile: MelodyProfile,
    transition: dict[str, Any],
) -> dict[str, Any]:
    """Density should equal primer_density + BiMMuDa delta, within 2σ."""
    delta = transition["delta_density_notes_per_bar"]
    expected = float(primer_profile.rhythmic_density) + float(delta["mean"])
    std = max(float(delta["std"]), 1e-6)
    observed = float(candidate_profile.rhythmic_density)
    z = abs(observed - expected) / std
    score = max(0.0, 1.0 - z / _CONDITIONAL_Z_MAX)
    return _check(
        "conditional_density",
        z <= _CONDITIONAL_Z_MAX,
        score,
        (
            f"conditional density {observed:.2f} differs from expected {expected:.2f} "
            f"(primer {primer_profile.rhythmic_density:.2f} + Δ {delta['mean']:+.2f}) "
            f"by {z:.2f}σ"
        ),
    )


def _conditional_pitch_span_check(
    notes: list[NoteEvent],
    primer_notes: list[NoteEvent],
    transition: dict[str, Any],
) -> dict[str, Any]:
    """Candidate span (max-min pitch) should equal primer_span + delta, within 2σ."""
    if not primer_notes:
        return _check("conditional_pitch_span", True, 1.0, "")
    primer_span = _pitch_span(primer_notes)
    candidate_span = _pitch_span(notes)
    delta = transition["delta_pitch_range_semitones"]
    expected = primer_span + float(delta["mean"])
    std = max(float(delta["std"]), 1e-6)
    z = abs(candidate_span - expected) / std
    score = max(0.0, 1.0 - z / _CONDITIONAL_Z_MAX)
    return _check(
        "conditional_pitch_span",
        z <= _CONDITIONAL_Z_MAX,
        score,
        (
            f"conditional span {candidate_span} st differs from expected {expected:.1f} st "
            f"(primer span {primer_span} + Δ {delta['mean']:+.2f}) by {z:.2f}σ"
        ),
    )


def _register_shift_check(
    notes: list[NoteEvent],
    primer_notes: list[NoteEvent],
    transition: dict[str, Any],
) -> dict[str, Any]:
    """Candidate mean pitch should equal primer mean pitch + delta_register, within 2σ."""
    if not primer_notes:
        return _check("register_shift", True, 1.0, "")
    primer_register = sum(int(note.pitch) for note in primer_notes) / len(primer_notes)
    candidate_register = sum(int(note.pitch) for note in notes) / len(notes)
    delta = transition["delta_register_semitones"]
    expected = primer_register + float(delta["mean"])
    std = max(float(delta["std"]), 1e-6)
    z = abs(candidate_register - expected) / std
    score = max(0.0, 1.0 - z / _CONDITIONAL_Z_MAX)
    return _check(
        "register_shift",
        z <= _CONDITIONAL_Z_MAX,
        score,
        (
            f"register {candidate_register:.1f} differs from expected {expected:.1f} "
            f"(primer {primer_register:.1f} + Δ {delta['mean']:+.2f}) by {z:.2f}σ"
        ),
    )


def _boundary_interval_check(
    notes: list[NoteEvent],
    primer_notes: list[NoteEvent],
    transition: dict[str, Any],
) -> dict[str, Any]:
    """|candidate_first - primer_last| should match BiMMuDa boundary magnitude, within 2σ."""
    if not primer_notes or not notes:
        return _check("boundary_interval", True, 1.0, "")
    primer_last = max(primer_notes, key=lambda note: note.onset + note.duration).pitch
    candidate_first = min(notes, key=lambda note: note.onset).pitch
    observed = abs(int(candidate_first) - int(primer_last))
    profile = transition["boundary_interval_semitones"]
    expected = float(profile["mean"])
    std = max(float(profile["std"]), 1e-6)
    z = abs(observed - expected) / std
    score = max(0.0, 1.0 - z / _CONDITIONAL_Z_MAX)
    return _check(
        "boundary_interval",
        z <= _CONDITIONAL_Z_MAX,
        score,
        (
            f"boundary interval |{candidate_first} - {primer_last}| = {observed} st "
            f"differs from expected {expected:.1f} st by {z:.2f}σ"
        ),
    )


def _pitch_span(notes: list[NoteEvent]) -> int:
    if not notes:
        return 0
    pitches = [int(note.pitch) for note in notes]
    return max(pitches) - min(pitches)


def _key_adherence_check(notes: list[NoteEvent], detected_key: str | None) -> dict[str, Any]:
    tonic, mode = parse_key(detected_key)
    scale = _scale_pitch_classes(tonic, mode)
    ratio = sum(1 for note in notes if note.pitch % 12 in scale) / len(notes)
    return _check(
        "key_adherence",
        ratio >= _KEY_ADHERENCE_MIN_RATIO,
        ratio,
        f"key adherence {ratio:.2f} below {_KEY_ADHERENCE_MIN_RATIO:.2f} for {detected_key or 'C major'}",
    )


def _pitch_range_check(notes: list[NoteEvent], primer_last_pitch: int | None) -> dict[str, Any]:
    if primer_last_pitch is None:
        return _check("pitch_range", True, 1.0, "")
    lower = primer_last_pitch - _PITCH_RANGE_SEMITONES
    upper = primer_last_pitch + _PITCH_RANGE_SEMITONES
    in_range = [lower <= note.pitch <= upper for note in notes]
    ratio = sum(in_range) / len(notes)
    return _check(
        "pitch_range",
        all(in_range),
        ratio,
        f"pitch range exceeds {lower}-{upper} around primer last pitch {primer_last_pitch}",
    )


def _density_check(candidate_profile: MelodyProfile, primer_profile: MelodyProfile) -> dict[str, Any]:
    primer_density = float(primer_profile.rhythmic_density)
    candidate_density = float(candidate_profile.rhythmic_density)
    if primer_density <= 0:
        return _check("rhythmic_density", True, 1.0, "")
    relative_delta = abs(candidate_density - primer_density) / primer_density
    score = max(0.0, 1.0 - relative_delta)
    return _check(
        "rhythmic_density",
        relative_delta <= _DENSITY_TOLERANCE,
        score,
        (
            f"rhythmic density {candidate_density:.2f} differs from primer "
            f"{primer_density:.2f} by {relative_delta:.0%}"
        ),
    )


def _chord_consonance_check(notes: list[NoteEvent], chord_symbols: list[str]) -> dict[str, Any]:
    if not chord_symbols:
        return _check("chord_consonance", True, 1.0, "")

    strong_notes = [note for note in notes if _is_strong_beat(note.onset)]
    if not strong_notes:
        return _check("chord_consonance", True, 1.0, "")

    consonant = 0
    checked = 0
    for note in strong_notes:
        # Assumes one chord per bar (4 beats in 4/4). All current ChordProgression
        # producers honour this convention; see the note on ChordProgression.chords
        # in shared/schemas.py before changing the math here.
        chord_index = int(math.floor(note.onset / _BEATS_PER_BAR)) % len(chord_symbols)
        chord_pcs = chord_symbol_to_pitch_classes(chord_symbols[chord_index])
        if not chord_pcs:
            continue
        checked += 1
        if note.pitch % 12 in chord_pcs:
            consonant += 1

    if checked == 0:
        return _check("chord_consonance", True, 1.0, "")
    ratio = consonant / checked
    return _check(
        "chord_consonance",
        ratio >= 0.5,
        ratio,
        f"strong-beat chord consonance {ratio:.2f} below 0.50",
    )


def chord_symbol_to_pitch_classes(symbol: str) -> set[int]:
    """Best-effort parser for common chord symbols used by HumMuse progressions."""
    symbol = (symbol or "").strip()
    if not symbol or symbol.upper() in {"N.C.", "NC", "REST"}:
        return set()

    root_match = re.match(r"^([A-G](?:#|b)?)(.*)$", symbol.split("/", 1)[0])
    if not root_match:
        return set()
    root = NAME_TO_PC.get(root_match.group(1))
    if root is None:
        return set()
    suffix = root_match.group(2).lower()

    template = _template_for_suffix(suffix)
    return {(root + interval) % 12 for interval in template.intervals}


def _template_for_suffix(suffix: str):
    if suffix.startswith("maj7") or suffix.startswith("ma7"):
        quality = "maj7"
    elif suffix.startswith("m7b5"):
        quality = "half-dim7"
    elif suffix.startswith("dim7"):
        quality = "dim7"
    elif suffix.startswith("dim"):
        quality = "dim"
    elif suffix.startswith("m7") or suffix.startswith("min7"):
        quality = "m7"
    elif suffix.startswith("m") or suffix.startswith("min"):
        quality = "minor"
    elif suffix.startswith("7"):
        quality = "7"
    elif suffix.startswith("sus2"):
        quality = "sus2"
    elif suffix.startswith("sus4") or suffix.startswith("sus"):
        quality = "sus4"
    elif suffix.startswith("aug") or suffix.startswith("+"):
        quality = "aug"
    elif suffix.startswith("6"):
        quality = "6"
    else:
        quality = "major"
    return next(template for template in QUALITY_TEMPLATES if template.quality == quality)


def _scale_pitch_classes(tonic: int, mode: str) -> set[int]:
    intervals = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    return {(tonic + interval) % 12 for interval in intervals}


def _last_pitch(notes: list[NoteEvent]) -> int | None:
    if not notes:
        return None
    return max(notes, key=lambda note: note.onset + note.duration).pitch


def _first_progression_symbols(chord_progressions: Iterable[ChordProgression]) -> list[str]:
    for progression in chord_progressions:
        if progression.chords:
            return list(progression.chords)
    return []


def _is_strong_beat(onset: float) -> bool:
    beat_in_bar = onset % _BEATS_PER_BAR
    return (
        abs(beat_in_bar - 0.0) <= _STRONG_BEAT_TOLERANCE
        or abs(beat_in_bar - 2.0) <= _STRONG_BEAT_TOLERANCE
    )


def _constraint_score(checks: list[dict[str, Any]]) -> float:
    if not checks:
        return 0.0
    return sum(float(check["score"]) for check in checks) / len(checks)


def _rejection_metadata(
    *,
    candidate_idx: int,
    model_id: Any,
    temperature: Any,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "model_id": str(model_id),
        "temperature": None if temperature is None else float(temperature),
        "rejection_reason": "; ".join(reason for reason in reasons if reason) or "unknown",
        "candidate_idx": int(candidate_idx),
    }


def _check(name: str, passed: bool, score: float, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "score": round(max(0.0, min(1.0, float(score))), 6),
        "reason": "" if passed else reason,
    }
