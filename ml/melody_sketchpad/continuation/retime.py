"""Post-hoc rhythm re-timer for Music Transformer continuation candidates.

Stretches a candidate's onsets so its onset density approaches the primer's
density. Two design decisions, both backed by the 2026-05-19 Antihero
diagnostic:

1. **Active-phrase density**: primer density is measured on the *first
   contiguous phrase* (notes up to the first inter-phrase rest of
   ``phrase_gap_beats`` or more), not the full primer. Pop verses with
   built-in pauses otherwise look artificially "slow" to the retimer —
   measuring the active phrase gives the listener-perceived tempo.

2. **Tight clip band** (``[0.7, 1.4]`` by default). Retiming should only
   apply *gentle* nudges. If a candidate's density is so off that it needs
   a 2× stretch to match, it should be rejected by the constraint filter,
   not rescued by uniform time-stretching — that path destroys rhythmic
   snap (notes become sustains) without restoring melodic correctness.

All inputs assumed to be in **beats** (matches ``NoteEvent`` conventions used
elsewhere in this package). Density is computed against wall-clock seconds
via the supplied BPM.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.schemas import NoteEvent

DEFAULT_PHRASE_GAP_BEATS = 1.5
DEFAULT_CLIP_LOW = 0.7
DEFAULT_CLIP_HIGH = 1.4
DEFAULT_DURATION_RELATIVE_TO_ONSET = 0.5


@dataclass(frozen=True)
class RetimeResult:
    """Outcome of a single retime pass — serialisable for run-log telemetry."""

    notes: list[NoteEvent]
    stretch_factor: float                # applied (post-clip)
    raw_stretch_factor: float            # computed (pre-clip)
    was_clipped: bool
    primer_active_density: float         # onsets/sec
    candidate_density_before: float      # onsets/sec
    candidate_density_after: float       # onsets/sec


def crop_to_first_phrase(
    notes: list[NoteEvent],
    *,
    phrase_gap_beats: float = DEFAULT_PHRASE_GAP_BEATS,
) -> list[NoteEvent]:
    """Return notes up to (not including) the first IOI gap >= ``phrase_gap_beats``.

    Notes are sorted by onset internally. A primer with no rest returns the
    full list. Single-note inputs return a copy.
    """
    if len(notes) < 2:
        return list(notes)
    sorted_notes = sorted(notes, key=lambda n: n.onset)
    out = [sorted_notes[0]]
    for prev, cur in zip(sorted_notes, sorted_notes[1:]):
        gap = float(cur.onset) - (float(prev.onset) + float(prev.duration))
        if gap >= phrase_gap_beats:
            break
        out.append(cur)
    return out


def active_phrase_density_onsets_per_sec(
    notes_beats: list[NoteEvent],
    *,
    bpm: float,
    phrase_gap_beats: float = DEFAULT_PHRASE_GAP_BEATS,
) -> float:
    """Onsets per second of the first contiguous phrase.

    Notes are in beats; conversion to seconds uses ``bpm``. An empty phrase
    or a zero-length phrase returns 0.0 (caller must guard against this for
    division in ratio math).
    """
    phrase = crop_to_first_phrase(notes_beats, phrase_gap_beats=phrase_gap_beats)
    if not phrase:
        return 0.0
    sorted_notes = sorted(phrase, key=lambda n: n.onset)
    last_end_beats = max(float(n.onset) + float(n.duration) for n in sorted_notes)
    if last_end_beats <= 0.0 or bpm <= 0.0:
        return 0.0
    dur_s = last_end_beats * 60.0 / bpm
    return len(sorted_notes) / dur_s


def retime_to_primer_density(
    candidate_notes_beats: list[NoteEvent],
    primer_notes_beats: list[NoteEvent],
    *,
    candidate_bpm: float,
    primer_bpm: float,
    phrase_gap_beats: float = DEFAULT_PHRASE_GAP_BEATS,
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
    duration_relative_to_onset: float = DEFAULT_DURATION_RELATIVE_TO_ONSET,
) -> RetimeResult:
    """Gentle pace correction toward primer's active-phrase density.

    ``duration_relative_to_onset`` mixes onset-only and uniform stretching:
      - 0.0 = onset-only (note durations untouched; preserves rhythmic snap)
      - 1.0 = uniform stretch (durations grow at the same rate as gaps)
      - 0.5 = compromise (durations grow at half the onset rate)

    Returns an unchanged candidate when either side has zero density (empty
    or degenerate input). The clip range is tight on purpose; candidates
    whose ``raw_stretch_factor`` falls outside it are almost certainly
    structurally wrong (not just paced wrong) and should be rejected by an
    upstream constraint filter, not rescued by stretching.
    """
    if not candidate_notes_beats:
        return RetimeResult([], 1.0, 1.0, False, 0.0, 0.0, 0.0)

    primer_density = active_phrase_density_onsets_per_sec(
        primer_notes_beats,
        bpm=primer_bpm,
        phrase_gap_beats=phrase_gap_beats,
    )
    candidate_density_before = active_phrase_density_onsets_per_sec(
        candidate_notes_beats,
        bpm=candidate_bpm,
        phrase_gap_beats=phrase_gap_beats,
    )

    if primer_density <= 0.0 or candidate_density_before <= 0.0:
        return RetimeResult(
            notes=list(candidate_notes_beats),
            stretch_factor=1.0,
            raw_stretch_factor=1.0,
            was_clipped=False,
            primer_active_density=primer_density,
            candidate_density_before=candidate_density_before,
            candidate_density_after=candidate_density_before,
        )

    # Density scales as 1 / (stretch on onset times). To make candidate's
    # onsets/sec match the primer's, multiply onsets by candidate/primer
    # density ratio. A denser candidate → ratio > 1 → slow down (longer gaps).
    raw_stretch = candidate_density_before / primer_density
    stretch = max(clip_low, min(clip_high, raw_stretch))
    was_clipped = stretch != raw_stretch

    duration_factor = 1.0 + (stretch - 1.0) * duration_relative_to_onset

    retimed = [
        NoteEvent(
            pitch=int(n.pitch),
            onset=float(n.onset) * stretch,
            duration=float(n.duration) * duration_factor,
            velocity=int(n.velocity),
            confidence=float(getattr(n, "confidence", 1.0)),
        )
        for n in candidate_notes_beats
    ]

    candidate_density_after = active_phrase_density_onsets_per_sec(
        retimed,
        bpm=candidate_bpm,
        phrase_gap_beats=phrase_gap_beats,
    )

    return RetimeResult(
        notes=retimed,
        stretch_factor=stretch,
        raw_stretch_factor=raw_stretch,
        was_clipped=was_clipped,
        primer_active_density=primer_density,
        candidate_density_before=candidate_density_before,
        candidate_density_after=candidate_density_after,
    )
