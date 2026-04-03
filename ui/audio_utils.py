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
    return base64.b64decode(value)


def midi_pitch_to_frequency(midi_pitch: int) -> float:
    return 440.0 * (2 ** ((midi_pitch - 69) / 12))


def synthesize_wave_from_notes(
    notes: list[dict[str, Any]],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    if not notes:
        raise ValueError("Cannot synthesize audio without note events")

    total_duration = max(float(note["onset"]) + float(note["duration"]) for note in notes) + 0.25
    total_samples = max(1, int(total_duration * sample_rate))
    pcm = np.zeros(total_samples, dtype=np.float64)

    for note in notes:
        pitch = int(note["pitch"])
        onset = float(note["onset"])
        duration = float(note["duration"])
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
        return synthesize_wave_from_notes(notes), "note_synth"

    if midi_bytes:
        raise RuntimeError(
            "MIDI data is available, but playback needs either note events or a configured SoundFont for midi2audio."
        )

    raise RuntimeError("No melody material is available for playback.")
