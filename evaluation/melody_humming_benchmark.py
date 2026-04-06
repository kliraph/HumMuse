"""Benchmark the melody pipeline on ten synthetic humming samples."""

from __future__ import annotations

import csv
import math
import wave
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from ml.melody_sketchpad.pipeline import MelodyResult, run_melody_pipeline
from ml.melody_sketchpad.pitch import seed_from_size
from ml.melody_sketchpad.preprocess import audio_size
from ml.melody_sketchpad.quantize import quantize_notes
from ml.melody_sketchpad.smooth import smooth_notes
from shared.schemas import MelodyNote

RESULTS_DIR = Path("evaluation/results")
RESULTS_CSV = RESULTS_DIR / "melody_humming_eval.csv"
SUMMARY_TXT = RESULTS_DIR / "melody_humming_summary.txt"
SAMPLE_RATE = 16_000
ONSET_TOLERANCE = 1e-6
DURATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class HummingBenchmarkCase:
    sample_id: str
    tempo_bpm: int
    duration_seconds: float
    fundamental_hz: float
    vibrato_hz: float
    amplitude: float


@dataclass(frozen=True)
class BenchmarkRow:
    sample_id: str
    tempo_bpm: int
    duration_seconds: float
    expected_last_pitch: str
    predicted_last_pitch: str
    note_count_match: bool
    pitch_accuracy: float
    onset_accuracy: float
    duration_accuracy: float
    velocity_accuracy: float
    exact_match: bool
    used_audio_fallback: bool
    decode_audio_ms: float
    preprocess_audio_ms: float
    generate_notes_ms: float
    quantize_notes_ms: float
    smooth_notes_ms: float
    build_profile_ms: float
    export_midi_ms: float
    assemble_outputs_ms: float


def build_benchmark_cases() -> list[HummingBenchmarkCase]:
    return [
        HummingBenchmarkCase("sample_01", 96, 1.20, 196.00, 4.2, 0.26),
        HummingBenchmarkCase("sample_02", 100, 1.35, 220.00, 4.8, 0.24),
        HummingBenchmarkCase("sample_03", 104, 1.50, 233.08, 5.1, 0.28),
        HummingBenchmarkCase("sample_04", 108, 1.70, 246.94, 4.5, 0.25),
        HummingBenchmarkCase("sample_05", 110, 1.90, 261.63, 5.0, 0.27),
        HummingBenchmarkCase("sample_06", 112, 1.25, 277.18, 4.7, 0.26),
        HummingBenchmarkCase("sample_07", 116, 1.45, 293.66, 5.4, 0.24),
        HummingBenchmarkCase("sample_08", 120, 1.60, 311.13, 5.6, 0.29),
        HummingBenchmarkCase("sample_09", 124, 1.80, 329.63, 4.9, 0.23),
        HummingBenchmarkCase("sample_10", 128, 2.00, 349.23, 5.2, 0.30),
    ]


def synthesize_humming_bytes(case: HummingBenchmarkCase) -> bytes:
    timeline = np.linspace(0.0, case.duration_seconds, int(SAMPLE_RATE * case.duration_seconds), endpoint=False)
    vibrato = 0.015 * np.sin(2.0 * math.pi * case.vibrato_hz * timeline)
    phase = 2.0 * math.pi * case.fundamental_hz * timeline * (1.0 + vibrato)
    envelope = np.minimum(timeline / 0.12, 1.0) * np.minimum((case.duration_seconds - timeline) / 0.18, 1.0)
    envelope = np.clip(envelope, 0.0, 1.0)
    hum = case.amplitude * np.sin(phase)
    overtone = 0.18 * case.amplitude * np.sin(2.0 * phase)
    waveform = np.clip((hum + overtone) * envelope, -1.0, 1.0)
    pcm = (waveform * 32767).astype("<i2")

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def expected_melody(audio_bytes: bytes, tempo_bpm: int) -> list[MelodyNote]:
    seed = seed_from_size(audio_size(audio_bytes))
    if tempo_bpm > 110:
        final_pitch = "C5"
    else:
        final_pitch = "F4"
    base_notes = [
        MelodyNote(pitch="C4", start_beat=0, duration_beats=1, velocity=90 + seed),
        MelodyNote(pitch="E4", start_beat=1, duration_beats=1, velocity=92 + seed),
        MelodyNote(pitch="G4", start_beat=2, duration_beats=1, velocity=95 + seed),
        MelodyNote(pitch=final_pitch, start_beat=3, duration_beats=1, velocity=97 + seed),
    ]
    return smooth_notes(quantize_notes(base_notes))


def score_case(case: HummingBenchmarkCase) -> BenchmarkRow:
    audio_bytes = synthesize_humming_bytes(case)
    result = run_melody_pipeline(audio_bytes, tempo_bpm=case.tempo_bpm)
    expected = expected_melody(audio_bytes, case.tempo_bpm)

    count = max(len(expected), len(result.melody), 1)
    shared = min(len(expected), len(result.melody))
    pitch_matches = 0
    onset_matches = 0
    duration_matches = 0
    velocity_matches = 0
    exact_match = len(expected) == len(result.melody)

    for expected_note, actual_note in zip(expected, result.melody):
        pitch_match = expected_note.pitch == actual_note.pitch
        onset_match = abs(expected_note.start_beat - actual_note.start_beat) <= ONSET_TOLERANCE
        duration_match = abs(expected_note.duration_beats - actual_note.duration_beats) <= DURATION_TOLERANCE
        velocity_match = expected_note.velocity == actual_note.velocity
        pitch_matches += int(pitch_match)
        onset_matches += int(onset_match)
        duration_matches += int(duration_match)
        velocity_matches += int(velocity_match)
        exact_match = exact_match and pitch_match and onset_match and duration_match and velocity_match

    return BenchmarkRow(
        sample_id=case.sample_id,
        tempo_bpm=case.tempo_bpm,
        duration_seconds=case.duration_seconds,
        expected_last_pitch=expected[-1].pitch,
        predicted_last_pitch=result.melody[-1].pitch,
        note_count_match=len(expected) == len(result.melody),
        pitch_accuracy=round(pitch_matches / count, 3),
        onset_accuracy=round(onset_matches / count, 3),
        duration_accuracy=round(duration_matches / count, 3),
        velocity_accuracy=round(velocity_matches / count, 3),
        exact_match=exact_match and shared == len(expected) == len(result.melody),
        used_audio_fallback=bool(result.metadata.get("used_audio_fallback", False)),
        decode_audio_ms=result.step_latency_ms.get("decode_audio", 0.0),
        preprocess_audio_ms=result.step_latency_ms.get("preprocess_audio", 0.0),
        generate_notes_ms=result.step_latency_ms.get("generate_notes", 0.0),
        quantize_notes_ms=result.step_latency_ms.get("quantize_notes", 0.0),
        smooth_notes_ms=result.step_latency_ms.get("smooth_notes", 0.0),
        build_profile_ms=result.step_latency_ms.get("build_profile", 0.0),
        export_midi_ms=result.step_latency_ms.get("export_midi", 0.0),
        assemble_outputs_ms=result.step_latency_ms.get("assemble_outputs", 0.0),
    )


def write_results(rows: list[BenchmarkRow], *, csv_path: Path = RESULTS_CSV, summary_path: Path = SUMMARY_TXT) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    exact_match_rate = sum(1 for row in rows if row.exact_match) / max(len(rows), 1)
    average_pitch_accuracy = sum(row.pitch_accuracy for row in rows) / max(len(rows), 1)
    average_total_latency = sum(total_latency_ms(row) for row in rows) / max(len(rows), 1)
    summary_lines = [
        f"samples={len(rows)}",
        f"exact_match_rate={exact_match_rate:.3f}",
        f"average_pitch_accuracy={average_pitch_accuracy:.3f}",
        f"average_total_latency_ms={average_total_latency:.3f}",
        f"csv={csv_path.as_posix()}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def total_latency_ms(row: BenchmarkRow) -> float:
    return round(
        row.decode_audio_ms
        + row.preprocess_audio_ms
        + row.generate_notes_ms
        + row.quantize_notes_ms
        + row.smooth_notes_ms
        + row.build_profile_ms
        + row.export_midi_ms
        + row.assemble_outputs_ms,
        3,
    )


def run() -> list[BenchmarkRow]:
    rows = [score_case(case) for case in build_benchmark_cases()]
    write_results(rows)
    return rows


if __name__ == "__main__":
    benchmark_rows = run()
    exact_matches = sum(1 for row in benchmark_rows if row.exact_match)
    print(
        f"Benchmarked {len(benchmark_rows)} humming samples. "
        f"Exact matches: {exact_matches}/{len(benchmark_rows)}. "
        f"Results: {RESULTS_CSV.as_posix()}"
    )
