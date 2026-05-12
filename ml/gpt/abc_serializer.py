"""Render SessionState as ABC notation for the SFT YandexGPT Lite explanation model.

The fine-tuned model was trained on ChatMusician's SFT corpus, which uses ABC
notation for score-grounded analysis tasks. Passing the session in ABC keeps
the model inside its training distribution. Assumptions: 4/4 meter,
monophonic melody, one chord per bar (matches `ChordProgression.chords`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from shared.schemas import ChordProgression, NoteEvent, SessionState

_PITCH_CLASS_TO_ABC: list[tuple[str, str]] = [
    ("C", ""), ("C", "^"), ("D", ""), ("D", "^"),
    ("E", ""), ("F", ""), ("F", "^"), ("G", ""),
    ("G", "^"), ("A", ""), ("A", "^"), ("B", ""),
]

DEFAULT_METER = "4/4"
DEFAULT_NOTE_LENGTH = "1/4"
DEFAULT_TEMPO_BPM = 120
DEFAULT_KEY = "Cmaj"
DEFAULT_TITLE = "HumMuse Session"
BEATS_PER_BAR = 4
MAX_BARS = 32

_KEY_QUALITY_MAP = {
    "major": "maj",
    "minor": "min",
    "maj": "maj",
    "min": "min",
    "dorian": "dor",
    "phrygian": "phr",
    "lydian": "lyd",
    "mixolydian": "mix",
    "aeolian": "min",
    "locrian": "loc",
}


@dataclass(frozen=True)
class ABCSerializationOptions:
    meter: str = DEFAULT_METER
    note_length: str = DEFAULT_NOTE_LENGTH
    title: str = DEFAULT_TITLE
    max_bars: int = MAX_BARS


def session_state_to_abc(
    session_state: SessionState,
    *,
    options: ABCSerializationOptions | None = None,
) -> str:
    """Render `session_state` as a single ABC notation string."""
    opts = options or ABCSerializationOptions()
    headers = _format_headers(session_state, opts)
    body = _format_body(session_state, opts)
    return f"{headers}\n{body}".rstrip() + "\n"


def midi_to_abc_pitch(midi_pitch: int) -> str:
    """Convert a MIDI pitch (0-127) into an ABC note token without duration."""
    if not 0 <= midi_pitch <= 127:
        raise ValueError(f"MIDI pitch out of range: {midi_pitch}")
    pitch_class = midi_pitch % 12
    octave = midi_pitch // 12 - 1
    letter, accidental = _PITCH_CLASS_TO_ABC[pitch_class]
    if octave >= 5:
        base = letter.lower()
        marks = "'" * (octave - 5)
    elif octave == 4:
        base = letter
        marks = ""
    else:
        base = letter
        marks = "," * (4 - octave)
    return f"{accidental}{base}{marks}"


def _format_headers(session_state: SessionState, opts: ABCSerializationOptions) -> str:
    tempo = (
        int(round(session_state.detected_tempo))
        if session_state.detected_tempo
        else DEFAULT_TEMPO_BPM
    )
    key = _normalize_key(session_state.detected_key) or DEFAULT_KEY
    return (
        "X:1\n"
        f"T:{opts.title}\n"
        f"M:{opts.meter}\n"
        f"L:{opts.note_length}\n"
        f"Q:1/4={tempo}\n"
        f"K:{key}"
    )


def _format_body(session_state: SessionState, opts: ABCSerializationOptions) -> str:
    chord_per_bar = _first_progression_chords(session_state.chord_progressions, opts.max_bars)
    bars = _group_notes_by_bar(session_state.melody_notes, opts.max_bars)
    if not bars and not chord_per_bar:
        return "% (empty session — no melody or chord progression yet)"
    bar_count = max(len(bars), len(chord_per_bar))
    lines: list[str] = []
    for bar_idx in range(bar_count):
        notes = bars[bar_idx] if bar_idx < len(bars) else []
        chord = chord_per_bar[bar_idx] if bar_idx < len(chord_per_bar) else None
        lines.append(_format_bar(notes, chord))
    return "\n".join(lines)


def _format_bar(notes: list[NoteEvent], chord: str | None) -> str:
    tokens: list[str] = []
    if chord:
        tokens.append(f"\"{chord}\"")
    if not notes:
        tokens.append(_format_rest(BEATS_PER_BAR))
    else:
        ordered = sorted(notes, key=lambda note: note.onset)
        for note in ordered:
            tokens.append(_format_note(note))
        played_beats = sum(min(note.duration, BEATS_PER_BAR) for note in ordered)
        remaining = BEATS_PER_BAR - played_beats
        if remaining >= 0.25:
            tokens.append(_format_rest(remaining))
    return " ".join(tokens) + " |"


def _format_note(note: NoteEvent) -> str:
    pitch = midi_to_abc_pitch(note.pitch)
    duration_token = _format_duration(min(note.duration, BEATS_PER_BAR))
    return f"{pitch}{duration_token}"


def _format_rest(beats: float) -> str:
    return f"z{_format_duration(beats)}"


def _format_duration(beats: float) -> str:
    if beats <= 0:
        return ""
    if abs(beats - round(beats)) < 1e-6:
        rounded = max(1, int(round(beats)))
        return "" if rounded == 1 else str(rounded)
    sixteenths = max(1, int(round(beats * 4)))
    if sixteenths % 4 == 0:
        return str(sixteenths // 4)
    if sixteenths == 1:
        return "/4"
    if sixteenths == 2:
        return "/2"
    if sixteenths == 3:
        return "3/4"
    return f"{sixteenths}/4"


def _group_notes_by_bar(notes: Iterable[NoteEvent], max_bars: int) -> list[list[NoteEvent]]:
    bars: list[list[NoteEvent]] = []
    for note in notes:
        bar_idx = int(note.onset // BEATS_PER_BAR)
        if bar_idx >= max_bars:
            continue
        while len(bars) <= bar_idx:
            bars.append([])
        bars[bar_idx].append(note)
    return bars


def _first_progression_chords(progressions: list[ChordProgression], max_bars: int) -> list[str]:
    if not progressions:
        return []
    return list(progressions[0].chords)[:max_bars]


def _normalize_key(detected_key: str | None) -> str | None:
    if not detected_key:
        return None
    tokens = detected_key.replace("-", " ").split()
    if not tokens:
        return None
    tonic = tokens[0]
    quality = "maj"
    if len(tokens) > 1:
        raw_quality = tokens[1].lower()
        quality = _KEY_QUALITY_MAP.get(raw_quality, raw_quality[:3])
    return f"{tonic}{quality}"
