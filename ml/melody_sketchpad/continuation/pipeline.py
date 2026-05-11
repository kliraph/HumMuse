"""End-to-end melody continuation pipeline."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None

from ml.melody_sketchpad.continuation.constraints import filter_candidates
from ml.melody_sketchpad.continuation.emotion_temperature import (
    generate_ensemble_with_emotion_temperature,
)
from ml.melody_sketchpad.continuation.inference import (
    EnsembleMember,
    EnsembleMelodyContinuationModel,
)
from ml.melody_sketchpad.continuation.primer import notes_to_prompt_midi
from ml.melody_sketchpad.continuation.scoring import score_candidates
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import MelodySuggestion, SessionState

LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 128
_DEFAULT_TOP_N = 3
_SINGLE_MODEL_CANDIDATES = 8
_DEFAULT_SINGLE_TEMPERATURES = (0.5, 0.7, 0.9)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "continuation.json"
_DEFAULT_CONTINUATION_CONFIG: dict[str, Any] = {
    "continuation": {
        "engine": "ensemble",
        "device": "cpu",
        "top_k": 40,
        "single": {
            "checkpoint": "storage/models/Lakh-MT/best_seed1337.pt",
            "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
            "model_id": "single",
            "count": _SINGLE_MODEL_CANDIDATES,
            "temperatures": list(_DEFAULT_SINGLE_TEMPERATURES),
        },
        "ensemble": {
            "members": [
                {
                    "checkpoint": "storage/models/Lakh-MT/best_seed7.pt",
                    "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
                    "model_id": "seed7",
                    "temperatures": [0.5, 0.7, 0.9],
                    "count": 3,
                },
                {
                    "checkpoint": "storage/models/Lakh-MT/best_seed42.pt",
                    "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
                    "model_id": "seed42",
                    "temperatures": [0.5, 0.7, 0.9],
                    "count": 3,
                },
                {
                    "checkpoint": "storage/models/Lakh-MT/best_seed1337.pt",
                    "tokenizer": "storage/models/Lakh-MT/tokenizer.json",
                    "model_id": "seed1337",
                    "temperatures": [0.7, 0.9],
                    "count": 2,
                },
            ]
        },
    }
}


def continue_melody(
    session_state: SessionState,
    *,
    generator: Any | None = None,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    top_n: int = _DEFAULT_TOP_N,
    out_dir: Path | str | None = None,
) -> list[MelodySuggestion]:
    """
    Continue the active session melody with Music Transformer candidates.

    Flow: primer adapter -> generation -> constraints -> profile scoring -> top N
    `MelodySuggestion`s. The function mutates `session_state.user_params` with
    constraint traces useful for later run logging.
    """
    if not session_state.melody_notes:
        raise ValueError("session_state.melody_notes is required for continuation")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    primer_profile = session_state.melody_profile or build_melody_profile(session_state.melody_notes)
    model = generator or get_default_continuation_model()

    temp_context = None
    if out_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="hummuse_continue_")
        work_dir = Path(temp_context.name)
    else:
        work_dir = Path(out_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        prompt_path = notes_to_prompt_midi(
            session_state.melody_notes,
            tempo=float(session_state.detected_tempo or 120.0),
            out_path=work_dir / "primer.mid",
        )
        candidates = _generate_candidates(
            model,
            prompt_path,
            emotion_vector=session_state.emotion_vector,
            max_new_tokens=max_new_tokens,
            out_dir=work_dir,
        )
        _log_pool_regime_warning(candidates)
        filtered = filter_candidates(
            candidates,
            primer_notes=session_state.melody_notes,
            primer_profile=primer_profile,
            detected_key=session_state.detected_key,
            chord_progressions=session_state.chord_progressions,
            max_survivors=max(top_n, 5),
        )
        scored = score_candidates(filtered.survivors, primer_profile)
        suggestions = [
            _candidate_to_suggestion(candidate)
            for candidate in scored[:top_n]
        ]

        session_state.user_params["continuation_constraint_trace"] = filtered.trace
        session_state.user_params["continuation_rejection_metadata"] = filtered.rejection_metadata
        session_state.user_params["continuation_candidate_count"] = len(candidates)
        session_state.user_params["continuation_survivor_count"] = len(filtered.survivors)
        return suggestions
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def load_continuation_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load continuation configuration.

    The tracked default lives at `config/continuation.json`. Local runs can
    override it with `HUMMUSE_CONTINUATION_CONFIG`; missing files fall back to
    the built-in single-model defaults.
    """
    if load_dotenv is not None:
        load_dotenv()

    config = deepcopy(_DEFAULT_CONTINUATION_CONFIG)
    selected_path = Path(config_path or os.environ.get("HUMMUSE_CONTINUATION_CONFIG", _DEFAULT_CONFIG_PATH))
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        _deep_update(config, loaded)
    elif config_path is not None or "HUMMUSE_CONTINUATION_CONFIG" in os.environ:
        raise FileNotFoundError(f"Continuation config not found: {selected_path}")
    return config


def get_default_continuation_model(config: dict[str, Any] | None = None) -> EnsembleMelodyContinuationModel:
    """Build the configured Lakh-MT continuation generator."""
    return build_continuation_model_from_config(config or load_continuation_config())


def build_continuation_model_from_config(config: dict[str, Any]) -> EnsembleMelodyContinuationModel:
    """
    Instantiate the configured continuation generator.

    Both `engine="single"` and `engine="ensemble"` produce an
    `EnsembleMelodyContinuationModel`; the single case is just a
    one-member ensemble. This keeps the downstream pipeline on a single
    code path while preserving the per-model_id bookkeeping that Phase 8
    ablation depends on.
    """
    continuation = config.get("continuation", config)
    engine = str(continuation.get("engine", "ensemble")).lower()
    device = continuation.get("device", "cpu")
    top_k = int(continuation.get("top_k", 40))

    if engine == "single":
        single = continuation.get("single", {})
        members = [_member_from_single_config(single)]
    elif engine == "ensemble":
        ensemble = continuation.get("ensemble", {})
        raw_members = _required(ensemble, "members", "continuation.ensemble.members")
        members = [_member_from_config(raw_member, index) for index, raw_member in enumerate(raw_members)]
    else:
        raise ValueError("continuation.engine must be 'single' or 'ensemble'")

    return EnsembleMelodyContinuationModel(members, device=device, top_k=top_k)


def _generate_candidates(
    generator: EnsembleMelodyContinuationModel,
    prompt_path: Path,
    *,
    emotion_vector,
    max_new_tokens: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    return generate_ensemble_with_emotion_temperature(
        generator,
        prompt_path,
        emotion_vector=emotion_vector,
        max_new_tokens=max_new_tokens,
        out_dir=out_dir,
    )


def _candidate_to_suggestion(candidate: dict[str, Any]) -> MelodySuggestion:
    midi_path = Path(candidate["midi_path"])
    constraint_text = candidate.get("constraint_explanation", "")
    score_breakdown = dict(candidate.get("score_breakdown", {}))
    model_id = candidate.get("model_id", "single")
    temperature = candidate.get("temperature")
    explanation = (
        f"Music Transformer continuation from {model_id} at T={temperature:.2f}. "
        f"{constraint_text} Profile score={float(candidate.get('profile_score', 0.0)):.3f}."
    )
    return MelodySuggestion(
        midi_bytes=midi_path.read_bytes(),
        notes=list(candidate.get("notes", [])),
        explanation=explanation,
        coherence_score=float(candidate.get("profile_score", 0.0)),
        engine="music_transformer",
        model_id=str(model_id) if model_id is not None else None,
        avg_log_prob=float(candidate.get("log_prob", 0.0)),
        constraint_trace=dict(candidate.get("constraint_trace", {})),
        score_breakdown=score_breakdown,
    )


def _log_pool_regime_warning(candidates: list[dict[str, Any]]) -> None:
    """
    Warn when the candidate pool has collapsed to a single sampling regime.

    A "regime" is a (model_id, temperature) pair. Two or more regimes are
    required for the constraint filter to do meaningful selection — if every
    candidate came from the same seed at the same temperature, the filter is
    just picking among near-duplicates. This check is independent of pool
    size: a 4-candidate single-model run with a 2-temperature schedule has
    2 regimes and is fine; an 8-candidate run that all collapsed to one
    regime is not.
    """
    if not candidates:
        return
    regimes = {
        (candidate.get("model_id", "single"), candidate.get("temperature"))
        for candidate in candidates
    }
    if len(regimes) < 2:
        LOGGER.warning(
            "continuation_candidate_pool_collapsed",
            extra={
                "candidate_count": len(candidates),
                "regime_count": len(regimes),
            },
        )


def _member_from_config(raw_member: dict[str, Any], index: int) -> EnsembleMember:
    label = f"continuation.ensemble.members[{index}]"
    return EnsembleMember(
        checkpoint_path=_resolve_config_path(_required(raw_member, "checkpoint", f"{label}.checkpoint")),
        tokenizer_path=_resolve_config_path(_required(raw_member, "tokenizer", f"{label}.tokenizer")),
        model_id=str(_required(raw_member, "model_id", f"{label}.model_id")),
        temperatures=_required(raw_member, "temperatures", f"{label}.temperatures"),
        count=_required(raw_member, "count", f"{label}.count"),
    )


def _member_from_single_config(single: dict[str, Any]) -> EnsembleMember:
    return EnsembleMember(
        checkpoint_path=_resolve_config_path(_required(single, "checkpoint", "continuation.single.checkpoint")),
        tokenizer_path=_resolve_config_path(_required(single, "tokenizer", "continuation.single.tokenizer")),
        model_id=str(single.get("model_id", "single")),
        temperatures=single.get("temperatures", _DEFAULT_SINGLE_TEMPERATURES),
        count=single.get("count", _SINGLE_MODEL_CANDIDATES),
    )


def _resolve_config_path(value: Path | str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _required(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config value: {label}")
    return mapping[key]


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target
