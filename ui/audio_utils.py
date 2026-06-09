"""Helpers for turning session melody material into playable audio."""

from __future__ import annotations

import base64
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency path
    from midi2audio import FluidSynth
except ModuleNotFoundError:  # pragma: no cover - covered through fallback tests
    FluidSynth = None


DEFAULT_SAMPLE_RATE = 22050


def decode_midi_bytes(value: str | bytes | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    # Pydantic v2 with ser_json_bytes="base64" emits URL-safe base64
    # WITHOUT padding, so restore the padding before decoding.
    padding = (-len(value)) % 4
    return base64.urlsafe_b64decode(value + ("=" * padding))


def midi_pitch_to_frequency(midi_pitch: int) -> float:
    return 440.0 * (2 ** ((midi_pitch - 69) / 12))


# Chord-symbol → MIDI mapping for audio preview. Kept intentionally
# minimal: only the qualities the DQN backend currently emits, plus a
# few common extras. Unknown qualities fall back to a major triad so
# the audio preview never breaks on an unfamiliar symbol — better a
# slightly wrong chord than a missing preview.
_NOTE_TO_SEMITONE: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}
_QUALITY_INTERVALS: dict[str, list[int]] = {
    "": [0, 4, 7],          # major
    "M": [0, 4, 7],
    "maj": [0, 4, 7],
    "m": [0, 3, 7],         # minor
    "min": [0, 3, 7],
    "-": [0, 3, 7],
    "aug": [0, 4, 8],       # augmented
    "+": [0, 4, 8],
    "dim": [0, 3, 6],       # diminished
    "°": [0, 3, 6],
    "o": [0, 3, 6],
    "7": [0, 4, 7, 10],     # dominant 7
    "M7": [0, 4, 7, 11],    # major 7
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],    # minor 7
    "m7b5": [0, 3, 6, 10],  # half-diminished
    "ø": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "°7": [0, 3, 6, 9],
    "6": [0, 4, 7, 9],      # major 6 (DQN emits this)
    "m6": [0, 3, 7, 9],     # minor 6 (DQN emits this)
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

# Quality prefixes checked longest-first when a full suffix doesn't match an
# exact entry above (e.g. extended/altered chords like "m7b9", "maj9",
# "sus4add2"). Picks the closest base quality instead of silently collapsing
# every unknown extension to a major triad. Order matters: longer/more-specific
# keys must precede their prefixes ("maj7" before "maj", "m7" before "m").
_QUALITY_PREFIXES: tuple[tuple[str, list[int]], ...] = (
    ("maj7", [0, 4, 7, 11]),
    ("dim7", [0, 3, 6, 9]),
    ("m7b5", [0, 3, 6, 10]),
    ("min", [0, 3, 7]),
    ("maj", [0, 4, 7]),
    ("sus2", [0, 2, 7]),
    ("sus4", [0, 5, 7]),
    ("sus", [0, 5, 7]),
    ("dim", [0, 3, 6]),
    ("aug", [0, 4, 8]),
    ("m7", [0, 3, 7, 10]),
    ("m6", [0, 3, 7, 9]),
    ("m", [0, 3, 7]),
    ("-", [0, 3, 7]),
    ("7", [0, 4, 7, 10]),
    ("6", [0, 4, 7, 9]),
    ("+", [0, 4, 8]),
    ("°", [0, 3, 6]),
    ("o", [0, 3, 6]),
    ("ø", [0, 3, 6, 10]),
)


def _quality_to_intervals(quality: str) -> list[int]:
    """Map a chord-quality suffix to intervals, degrading gracefully.

    Tries an exact match first, then the longest quality prefix (so
    ``"m7b9"`` reads as a minor-7 rather than a major triad). Falls back to a
    major triad only when nothing matches at all.
    """
    if quality in _QUALITY_INTERVALS:
        return _QUALITY_INTERVALS[quality]
    # "maj9"/"maj11"/"maj13" etc. carry the major 7th — preview as maj7.
    if quality.startswith("maj") and quality[3:4].isdigit():
        return [0, 4, 7, 11]
    # A bare extension number ("9", "11", "13") implies a dominant 7th stack.
    # ("6"/"69" are handled by exact-match / prefix above.)
    if quality[0:1].isdigit():
        return [0, 4, 7, 10]
    for prefix, intervals in _QUALITY_PREFIXES:
        if quality.startswith(prefix):
            return intervals
    return [0, 4, 7]


def parse_chord_symbol(symbol: str) -> tuple[int, list[int]]:
    """Return ``(root_semitone, intervals_from_root)`` for a chord name.

    Examples: ``"C"`` → ``(0, [0, 4, 7])``; ``"Caug"`` → ``(0, [0, 4, 8])``;
    ``"F#m7"`` → ``(6, [0, 3, 7, 10])``. Slash chords keep the upper
    structure and add the bass note as the lowest voice
    (``"C/E"`` → ``(0, [-8, 0, 4, 7])``). Unknown extensions degrade to the
    closest base quality (``"Dm7b9"`` → minor-7) rather than a bare major
    triad, so the audio preview reflects the chord's actual color instead of
    silently misrepresenting it.
    """
    if not symbol:
        return 0, [0, 4, 7]

    # Slash chord: "<chord>/<bass>". Parse the chord part, then drop the bass
    # an octave below the root so the inversion is audible in the preview.
    chord_part, _, bass_part = symbol.partition("/")
    if not chord_part:  # leading-slash garbage; nothing to voice
        return 0, [0, 4, 7]

    root_semitone, remainder = _split_root(chord_part)
    intervals = list(_quality_to_intervals(remainder))

    if bass_part:
        bass_semitone, _ = _split_root(bass_part)
        bass_pc_offset = (bass_semitone - root_semitone) % 12
        # A slash bass equal to the root is a no-op inversion — skip it.
        if bass_pc_offset != 0:
            # Voice the bass in the octave below the chord root.
            bass_interval = bass_pc_offset - 12
            if bass_interval not in intervals:
                intervals = [bass_interval, *intervals]

    return root_semitone, intervals


def _split_root(token: str) -> tuple[int, str]:
    """Split a chord token into ``(root_semitone, quality_remainder)``."""
    root_str = token[0].upper()
    remainder = token[1:]
    if remainder and remainder[0] in ("#", "b"):
        root_str += remainder[0]
        remainder = remainder[1:]
    return _NOTE_TO_SEMITONE.get(root_str, 0), remainder


def synthesize_progression_audio(
    chords: list[str],
    *,
    bpm: float = 100.0,
    beats_per_chord: float = 2.0,
    root_octave: int = 4,
    deduplicate_consecutive: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Render a chord progression as a short WAV preview.

    Each chord plays for ``beats_per_chord`` beats at the given ``bpm``,
    voiced as a simple block triad/seventh starting at ``root_octave``.
    Consecutive duplicates are collapsed by default — a card showing
    "Caug ×18" should preview the *sound* of Caug once, not 40 seconds
    of the same chord. Re-uses ``synthesize_wave_from_notes`` for the
    actual signal generation so timbre matches the melody preview.
    """
    if not chords:
        raise ValueError("Cannot synthesize audio without chords")

    sequence = list(chords)
    if deduplicate_consecutive:
        deduped: list[str] = []
        for chord in sequence:
            if not deduped or deduped[-1] != chord:
                deduped.append(chord)
        sequence = deduped

    seconds_per_beat = 60.0 / max(1.0, float(bpm))
    chord_duration_s = float(beats_per_chord) * seconds_per_beat

    note_events: list[dict[str, Any]] = []
    for position, symbol in enumerate(sequence):
        root_semitone, intervals = parse_chord_symbol(symbol)
        onset = position * chord_duration_s
        for interval in intervals:
            # MIDI pitch = 12 * (octave + 1) + pitch class. Octave 4
            # places C at MIDI 60 (middle C) which sits cleanly under
            # a melody preview voiced an octave higher.
            midi_pitch = 12 * (root_octave + 1) + root_semitone + interval
            note_events.append(
                {
                    "pitch": midi_pitch,
                    "onset": onset,
                    "duration": chord_duration_s * 0.95,  # slight release gap
                    "velocity": 76,
                }
            )

    return synthesize_wave_from_notes(note_events, sample_rate=sample_rate)


def synthesize_wave_from_notes(
    notes: list[dict[str, Any]],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bpm: float | None = None,
) -> bytes:
    """Synthesize a sine-wave preview from note events.

    ``onset``/``duration`` are interpreted as **seconds** when ``bpm`` is None,
    or as **beats** (converted via the given tempo) when ``bpm`` is provided.
    The pipeline's ``NoteEvent``s are in beats (see ml.melody_sketchpad), so
    melody and continuation previews must pass ``bpm``; the chord-progression
    helper builds seconds directly and omits it.
    """
    if not notes:
        raise ValueError("Cannot synthesize audio without note events")

    time_scale = (60.0 / max(1.0, float(bpm))) if bpm else 1.0

    total_duration = (
        max(float(note["onset"]) + float(note["duration"]) for note in notes) * time_scale + 0.25
    )
    total_samples = max(1, int(total_duration * sample_rate))
    pcm = np.zeros(total_samples, dtype=np.float64)

    for note in notes:
        pitch = int(note["pitch"])
        onset = float(note["onset"]) * time_scale
        duration = float(note["duration"]) * time_scale
        velocity = int(note.get("velocity", 96))
        amplitude = min(1.0, max(0.1, velocity / 127.0)) * 0.25
        start = max(0, int(onset * sample_rate))
        end = min(total_samples, int((onset + duration) * sample_rate))
        frequency = midi_pitch_to_frequency(pitch)
        sample_count = end - start
        if sample_count <= 0:
            continue

        offsets = np.arange(sample_count, dtype=np.float64)
        elapsed = offsets / sample_rate
        progress = offsets / sample_count
        envelope = np.minimum(progress * 4.0, 1.0) * np.minimum((1.0 - progress) * 4.0, 1.0)
        pcm[start:end] += np.sin(2.0 * np.pi * frequency * elapsed) * amplitude * envelope

    peak = max(float(np.max(np.abs(pcm))), 1e-9)
    scale = 32767 / peak * 0.8
    frame_values = np.clip(pcm * scale, -32767, 32767).astype("<i2")

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frame_values.tobytes())
    return buffer.getvalue()


def render_audio_from_session(
    state: dict[str, Any],
    *,
    soundfont_path: str | None = None,
) -> tuple[bytes, str]:
    midi_bytes = decode_midi_bytes(state.get("melody_midi"))
    notes = state.get("melody_notes", [])

    if FluidSynth is not None and midi_bytes and soundfont_path:
        soundfont = Path(soundfont_path)
        if soundfont.exists():
            with tempfile.TemporaryDirectory() as temp_dir:
                midi_path = Path(temp_dir) / "melody.mid"
                wav_path = Path(temp_dir) / "melody.wav"
                midi_path.write_bytes(midi_bytes)
                synthesizer = FluidSynth(sound_font=str(soundfont))
                synthesizer.midi_to_audio(str(midi_path), str(wav_path))
                return wav_path.read_bytes(), "midi2audio"

    if notes:
        bpm = float(state.get("detected_tempo") or 100.0)
        return synthesize_wave_from_notes(notes, bpm=bpm), "note_synth"

    if midi_bytes:
        raise RuntimeError(
            "MIDI data is available, but playback needs either note events or a configured SoundFont for midi2audio."
        )

    raise RuntimeError("No melody material is available for playback.")
