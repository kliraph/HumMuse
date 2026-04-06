"""Tests for the humming benchmark harness."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from evaluation.melody_humming_benchmark import (
    BenchmarkRow,
    build_benchmark_cases,
    run,
    score_case,
    synthesize_humming_bytes,
    write_results,
)


def test_build_benchmark_cases_returns_ten_samples() -> None:
    cases = build_benchmark_cases()

    assert len(cases) == 10
    assert cases[0].sample_id == "sample_01"
    assert cases[-1].sample_id == "sample_10"


def test_synthesize_humming_bytes_returns_wav_payload() -> None:
    audio_bytes = synthesize_humming_bytes(build_benchmark_cases()[0])

    assert audio_bytes[:4] == b"RIFF"
    assert b"WAVE" in audio_bytes[:16]


def test_score_case_reports_exact_match_for_reference_sample() -> None:
    row = score_case(build_benchmark_cases()[0])

    assert isinstance(row, BenchmarkRow)
    assert row.note_count_match is True
    assert row.pitch_accuracy == 1.0
    assert row.onset_accuracy == 1.0
    assert row.duration_accuracy == 1.0
    assert row.velocity_accuracy == 1.0
    assert row.exact_match is True
    assert row.used_audio_fallback is False


def test_write_results_emits_csv_and_summary() -> None:
    rows = [score_case(build_benchmark_cases()[0])]
    with tempfile.TemporaryDirectory(dir="storage") as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "results.csv"
        summary_path = tmp_path / "summary.txt"

        write_results(rows, csv_path=csv_path, summary_path=summary_path)

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            written_rows = list(csv.DictReader(handle))
        summary_text = summary_path.read_text(encoding="utf-8")

        assert len(written_rows) == 1
        assert written_rows[0]["sample_id"] == "sample_01"
        assert "exact_match_rate=1.000" in summary_text
        assert "average_pitch_accuracy=1.000" in summary_text


def test_run_generates_ten_rows() -> None:
    rows = run()

    assert len(rows) == 10
