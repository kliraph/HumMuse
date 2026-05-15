"""Tests for backend.refinement_executor."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

import backend.refinement_executor as executor_mod
from backend.refinement_executor import (
    OperationResult,
    RefinementExecution,
    execute_refinement_plan,
)
from ml.gpt.pipeline import GPTPipelineResult
from shared.schemas import (
    ChordProgression,
    EmotionVector,
    MelodySuggestion,
    NoteEvent,
    RefinementOp,
    RefinementPlan,
    SessionState,
)


def _session(*, with_melody: bool = False, with_emotion: bool = True) -> SessionState:
    state = SessionState(session_id=uuid4())
    if with_emotion:
        state.emotion_vector = EmotionVector(valence=0.0, arousal=0.4)
    if with_melody:
        state.melody_notes = [
            NoteEvent(pitch=60, onset=0.0, duration=1.0, velocity=96, confidence=0.95),
            NoteEvent(pitch=64, onset=1.0, duration=1.0, velocity=96, confidence=0.95),
            NoteEvent(pitch=67, onset=2.0, duration=1.0, velocity=96, confidence=0.95),
        ]
        state.detected_key = "C major"
        state.detected_tempo = 120.0
    return state


def _plan(*ops: RefinementOp, interpretation: str = "test") -> RefinementPlan:
    return RefinementPlan(operations=list(ops), interpretation=interpretation)


def _harmonizer_op(**params: Any) -> RefinementOp:
    return RefinementOp(target="harmonizer", params=params, rationale="test-harmony")


def _melody_op(**params: Any) -> RefinementOp:
    return RefinementOp(target="melody_generator", params=params, rationale="test-melody")


def _lyric_op(**params: Any) -> RefinementOp:
    return RefinementOp(target="lyric_generator", params=params, rationale="test-lyric")


def _session_op(**params: Any) -> RefinementOp:
    return RefinementOp(target="session", params=params, rationale="test-session")


# --- harmonizer ---------------------------------------------------------------


def test_harmonizer_op_skips_when_no_melody() -> None:
    state = _session(with_melody=False)
    execution = execute_refinement_plan(state, _plan(_harmonizer_op(emotion_valence=-0.4)))
    [result] = execution.results
    assert result.status == "skipped"
    assert result.error == "no_melody"
    assert state.chord_progressions == []


def test_harmonizer_op_sets_absolute_valence_and_reruns_dqn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_dqn(notes, *, key, emotion_vector, top_k, tempo_bpm, **kwargs):
        captured["emotion_vector"] = emotion_vector
        captured["key"] = key
        captured["top_k"] = top_k
        captured["tempo_bpm"] = tempo_bpm
        return [ChordProgression(chords=["Am", "F", "C", "G"], score=0.9, explanation="darker")]

    monkeypatch.setattr(executor_mod, "generate_dqn_chords", fake_dqn)

    state = _session(with_melody=True)
    execution = execute_refinement_plan(state, _plan(_harmonizer_op(emotion_valence=-0.4)))

    [result] = execution.results
    assert result.status == "applied"
    assert captured["emotion_vector"].valence == pytest.approx(-0.4)
    assert state.emotion_vector.valence == pytest.approx(-0.4)
    assert state.chord_progressions[0].chords == ["Am", "F", "C", "G"]
    assert result.changes["progression_count"] == 1
    assert result.changes["top_progression"] == ["Am", "F", "C", "G"]


def test_harmonizer_valence_delta_is_relative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "generate_dqn_chords",
        lambda *a, **kw: [ChordProgression(chords=["C"], score=1.0, explanation="x")],
    )
    state = _session(with_melody=True)
    state.emotion_vector = EmotionVector(valence=-0.5, arousal=0.4)

    execute_refinement_plan(state, _plan(_harmonizer_op(emotion_valence_delta=0.25)))

    assert state.emotion_vector.valence == pytest.approx(-0.25)


def test_harmonizer_prefer_minor_color_floors_valence_to_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "generate_dqn_chords",
        lambda *a, **kw: [ChordProgression(chords=["C"], score=1.0, explanation="x")],
    )
    state = _session(with_melody=True)
    state.emotion_vector = EmotionVector(valence=0.5, arousal=0.4)

    execute_refinement_plan(state, _plan(_harmonizer_op(prefer_minor_color=True)))

    assert state.emotion_vector.valence <= -0.2


def test_harmonizer_reduce_minor_bias_raises_valence_to_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "generate_dqn_chords",
        lambda *a, **kw: [ChordProgression(chords=["C"], score=1.0, explanation="x")],
    )
    state = _session(with_melody=True)
    state.emotion_vector = EmotionVector(valence=-0.5, arousal=0.4)

    execute_refinement_plan(state, _plan(_harmonizer_op(reduce_minor_bias=True)))

    assert state.emotion_vector.valence >= 0.2


def test_harmonizer_records_unsupported_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "generate_dqn_chords",
        lambda *a, **kw: [ChordProgression(chords=["C"], score=1.0, explanation="x")],
    )
    state = _session(with_melody=True)

    execution = execute_refinement_plan(
        state,
        _plan(_harmonizer_op(chord_complexity="high", extensions=["7ths", "9ths"], section="chorus")),
    )

    [result] = execution.results
    assert "chord_complexity" in result.changes["unsupported_params"]
    assert "extensions" in result.changes["unsupported_params"]
    assert "section" in result.changes["unsupported_params"]


def test_harmonizer_clips_valence_to_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "generate_dqn_chords",
        lambda *a, **kw: [ChordProgression(chords=["C"], score=1.0, explanation="x")],
    )
    state = _session(with_melody=True)
    state.emotion_vector = EmotionVector(valence=0.9, arousal=0.4)

    execute_refinement_plan(state, _plan(_harmonizer_op(emotion_valence_delta=0.9)))

    assert state.emotion_vector.valence == pytest.approx(1.0)


# --- melody_generator ---------------------------------------------------------


def test_melody_op_skips_when_no_melody() -> None:
    state = _session(with_melody=False)
    execution = execute_refinement_plan(state, _plan(_melody_op(length="shorter")))
    [result] = execution.results
    assert result.status == "skipped"
    assert result.error == "no_melody"


def test_melody_op_applies_length_to_max_new_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_continue(state, *, top_n, max_new_tokens, **_):
        captured["top_n"] = top_n
        captured["max_new_tokens"] = max_new_tokens
        return [
            MelodySuggestion(
                midi_bytes=b"\x00",
                notes=[NoteEvent(pitch=60, onset=0, duration=1, velocity=80, confidence=0.9)],
                explanation="ok",
                coherence_score=0.8,
            )
        ]

    monkeypatch.setattr(executor_mod, "run_continuation_pipeline", fake_continue)

    state = _session(with_melody=True)
    execution = execute_refinement_plan(
        state,
        _plan(_melody_op(length="shorter", contour="rising", smooth_contour=True)),
    )

    [result] = execution.results
    assert result.status == "applied"
    assert captured["max_new_tokens"] == 64
    assert state.melody_suggestions and state.melody_suggestions[0].explanation == "ok"
    assert state.user_params["last_melody_shaping"]["length"] == "shorter"
    assert "contour" in result.changes["unsupported_params"]
    assert "smooth_contour" in result.changes["unsupported_params"]


def test_melody_op_longer_uses_larger_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_continue(state, *, top_n, max_new_tokens, **_):
        captured["max_new_tokens"] = max_new_tokens
        return []

    monkeypatch.setattr(executor_mod, "run_continuation_pipeline", fake_continue)

    state = _session(with_melody=True)
    execute_refinement_plan(state, _plan(_melody_op(length="longer")))
    assert captured["max_new_tokens"] == 192


# --- lyric_generator ----------------------------------------------------------


class _FakeLyricPipeline:
    def __init__(self, *, parsed: Any) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, Any]] = []

    def route_request(self, use_case: str, state: SessionState, instruction: str, **kwargs: Any) -> GPTPipelineResult:
        self.calls.append({"use_case": use_case, "instruction": instruction, **kwargs})
        return GPTPipelineResult(
            use_case=use_case,
            content=json.dumps(self.parsed) if isinstance(self.parsed, dict) else str(self.parsed),
            parsed=self.parsed,
            cache_hit=False,
            provider="yandex",
            model_version="test-model",
            latency_ms=20.0,
            prompt_token_count=50,
            completion_token_count=15,
            eval_mode=False,
        )


def test_lyric_op_falls_back_to_mock_when_no_pipeline() -> None:
    """Lyric op delegates to lyric_responder, which falls back to the mock
    builder when no GPT pipeline is available. The op is therefore always
    `applied`; the `source="fallback"` flag in changes records that the user
    got mock-generated suggestions rather than GPT output, and `error`
    records the reason.
    """
    state = _session()
    execution = execute_refinement_plan(state, _plan(_lyric_op(imagery="moonlight")), pipeline=None)
    [result] = execution.results
    assert result.status == "applied"
    assert result.error == "no_gpt_pipeline"
    assert result.changes["source"] == "fallback"
    # Mock builder produces non-empty suggestions for downstream UI.
    assert state.lyric_suggestions


def test_lyric_op_falls_back_to_mock_on_transport_error() -> None:
    """The responder also catches a raising pipeline and falls back to mock."""
    class _BoomPipeline:
        def route_request(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    state = _session()
    execution = execute_refinement_plan(
        state,
        _plan(_lyric_op(imagery="midnight", num_options=2)),
        pipeline=_BoomPipeline(),
    )
    [result] = execution.results
    assert result.status == "applied"
    assert result.changes["source"] == "fallback"
    assert result.error == "transport:RuntimeError"
    assert state.lyric_suggestions  # mock fills in


def test_lyric_op_applies_imagery_and_stores_suggestions() -> None:
    pipeline = _FakeLyricPipeline(parsed={
        "mode": "Poetic",
        "lyrics": ["Under the silver tide", "We hold the morning close"],
        "syllable_counts": [6, 7],
        "rationale": "test",
    })
    state = _session()

    execution = execute_refinement_plan(
        state,
        _plan(_lyric_op(imagery="silver tide", num_options=2, language="en")),
        pipeline=pipeline,
    )

    [result] = execution.results
    assert result.status == "applied"
    assert len(state.lyric_suggestions) == 2
    assert state.lyric_suggestions[0].text == "Under the silver tide"
    assert state.lyric_suggestions[0].syllable_count == 6
    assert state.lyric_suggestions[0].mode == "Poetic"
    assert pipeline.calls[0]["use_case"] == "lyric"
    assert pipeline.calls[0]["num_suggestions"] == 2
    assert "silver tide" in pipeline.calls[0]["instruction"]
    assert "English" in pipeline.calls[0]["instruction"]


def test_lyric_op_handles_string_parsed_payload() -> None:
    pipeline = _FakeLyricPipeline(parsed='{"mode": "Poetic", "lyrics": ["Light at dawn"], "syllable_counts": [3]}')
    state = _session()

    execute_refinement_plan(state, _plan(_lyric_op(num_options=1)), pipeline=pipeline)

    assert len(state.lyric_suggestions) == 1
    assert state.lyric_suggestions[0].text == "Light at dawn"


def test_lyric_op_russian_language_in_instruction() -> None:
    pipeline = _FakeLyricPipeline(parsed={"mode": "Poetic", "lyrics": ["В тёмной ночи"], "syllable_counts": [4]})
    state = _session()

    execute_refinement_plan(state, _plan(_lyric_op(num_options=1, language="ru")), pipeline=pipeline)

    assert "Russian" in pipeline.calls[0]["instruction"]


# --- session ------------------------------------------------------------------


def test_session_op_merges_preserve_and_genre() -> None:
    state = _session()

    execute_refinement_plan(
        state,
        _plan(_session_op(preserve=["melody"], genre="indie pop")),
    )

    assert state.user_params["preserve"] == ["melody"]
    assert state.user_params["genre"] == "indie pop"


def test_session_op_mood_updates_emotion_vector() -> None:
    state = _session(with_emotion=False)

    execute_refinement_plan(state, _plan(_session_op(mood="melancholic")))

    assert state.user_params["mood"] == "melancholic"
    assert state.emotion_vector is not None
    assert state.emotion_vector.valence < 0


# --- multi-op / failure handling ----------------------------------------------


def test_multi_op_plan_executes_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order: list[str] = []

    def fake_dqn(*a, **kw):
        call_order.append("harmonizer")
        return [ChordProgression(chords=["C"], score=1.0, explanation="x")]

    def fake_continue(state, *, top_n, max_new_tokens, **_):
        call_order.append("melody")
        return []

    monkeypatch.setattr(executor_mod, "generate_dqn_chords", fake_dqn)
    monkeypatch.setattr(executor_mod, "run_continuation_pipeline", fake_continue)

    state = _session(with_melody=True)
    execution = execute_refinement_plan(
        state,
        _plan(
            _harmonizer_op(emotion_valence=-0.3),
            _melody_op(length="shorter"),
            _session_op(genre="indie folk"),
        ),
    )

    assert call_order == ["harmonizer", "melody"]
    assert [r.target for r in execution.results] == ["harmonizer", "melody_generator", "session"]
    assert all(r.status == "applied" for r in execution.results)
    assert execution.applied_count == 3


def test_failing_op_does_not_block_subsequent_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_dqn(*a, **kw):
        raise RuntimeError("dqn checkpoint missing")

    monkeypatch.setattr(executor_mod, "generate_dqn_chords", fake_dqn)

    state = _session(with_melody=True)
    execution = execute_refinement_plan(
        state,
        _plan(_harmonizer_op(emotion_valence=-0.4), _session_op(genre="ambient")),
    )

    assert execution.results[0].status == "failed"
    assert "dqn checkpoint missing" in execution.results[0].error
    assert execution.results[1].status == "applied"
    assert state.user_params["genre"] == "ambient"


def test_execution_to_dict_serialisable() -> None:
    execution = RefinementExecution(
        results=[OperationResult(target="session", status="applied", rationale="x", changes={"k": 1})]
    )
    payload = execution.to_dict()
    json.dumps(payload)  # must serialise cleanly
    assert payload["results"][0]["target"] == "session"
