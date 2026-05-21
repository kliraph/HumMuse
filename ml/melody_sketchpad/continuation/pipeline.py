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

from ml.melody_sketchpad.continuation.constraints import decode_candidate_midi, filter_candidates
from ml.melody_sketchpad.continuation.emotion_temperature import (
    generate_ensemble_with_emotion_temperature,
)
from ml.melody_sketchpad.continuation.inference import (
    EnsembleMember,
    EnsembleMelodyContinuationModel,
)
from ml.melody_sketchpad.continuation.primer import notes_to_prompt_midi
from ml.melody_sketchpad.continuation.priors import load_transition_profile
from ml.melody_sketchpad.continuation.retime import retime_to_primer_density
from ml.melody_sketchpad.continuation.scoring import score_candidates
from ml.melody_sketchpad.profile import build_melody_profile
from shared.schemas import MelodySuggestion, NoteEvent, SessionState

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
                    "temperatures": [0.5, 0.7, 0.9],
                    "count": 3,
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
    retime_to_primer: bool = True,
) -> list[MelodySuggestion]:
    """
    Continue the active session melody with Music Transformer candidates.

    Flow: primer adapter -> generation -> constraints -> profile scoring ->
    *active-phrase retime* -> top N `MelodySuggestion`s. The function mutates
    `session_state.user_params` with constraint traces useful for later run
    logging.

    `retime_to_primer=True` (default) runs each surviving candidate through
    ``retime_to_primer_density`` — a gentle onset stretch toward the
    primer's active-phrase density (clipped to [0.7, 1.4]). Set to False to
    keep the raw Music Transformer timing.
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

        # Retime BEFORE filtering. The density gate inside filter_candidates
        # would otherwise reject candidates whose only problem is pacing —
        # which is exactly what retiming was designed to fix. Doing it first
        # gives the gate the corrected timing to evaluate against.
        retime_telemetry: list[dict[str, Any]] = []
        if retime_to_primer and candidates:
            retime_telemetry = _retime_all_candidates(
                candidates,
                primer_notes=session_state.melody_notes,
                primer_bpm=float(session_state.detected_tempo or 120.0),
            )

        filtered = filter_candidates(
            candidates,
            primer_notes=session_state.melody_notes,
            primer_profile=primer_profile,
            detected_key=session_state.detected_key,
            chord_progressions=session_state.chord_progressions,
            max_survivors=max(top_n, 5),
            primer_section=session_state.primer_section,
            target_section=session_state.target_section,
        )
        # Resolve the transition once more for the scoring stage. LRU-cached
        # in priors.py so the duplicate lookup is free. Passing the resolved
        # transition (vs. section labels) keeps score_candidates ignorant of
        # the JSON file layout.
        scoring_transition = load_transition_profile(
            session_state.primer_section,
            session_state.target_section,
        )
        scored = score_candidates(
            filtered.survivors,
            primer_profile,
            primer_notes=session_state.melody_notes,
            transition=scoring_transition,
        )
        top = scored[:top_n]

        suggestions = [
            _candidate_to_suggestion(candidate)
            for candidate in top
        ]

        session_state.user_params["continuation_constraint_trace"] = filtered.trace
        session_state.user_params["continuation_rejection_metadata"] = filtered.rejection_metadata
        session_state.user_params["continuation_candidate_count"] = len(candidates)
        session_state.user_params["continuation_survivor_count"] = len(filtered.survivors)
        if retime_to_primer:
            session_state.user_params["continuation_retime_trace"] = retime_telemetry
            session_state.user_params["continuation_retime_clipped_count"] = sum(
                1 for r in retime_telemetry if r.get("was_clipped")
            )
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


def _retime_all_candidates(
    candidates: list[dict[str, Any]],
    *,
    primer_notes: list[NoteEvent],
    primer_bpm: float,
) -> list[dict[str, Any]]:
    """Apply the active-phrase / tight-clip retimer to every candidate
    *before* the constraint filter runs.

    For each candidate this:
      - decodes the MIDI at ``candidate["midi_path"]`` to get its notes
      - calls ``retime_to_primer_density`` against the primer's active phrase
      - rewrites the MIDI on disk if the stretch factor isn't ~1.0 (so the
        downstream constraint filter — which re-decodes from disk — sees the
        retimed timing, and so does the rendered audio)
      - stashes the retime result on the candidate dict so
        ``_candidate_to_suggestion`` can surface it via ``score_breakdown``

    Per-candidate failures (bad MIDI, decode errors) are swallowed and the
    candidate is left untouched; the constraint filter will handle it.
    """
    telemetry: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        midi_path_str = candidate.get("midi_path")
        if not midi_path_str:
            continue
        midi_path = Path(midi_path_str)
        try:
            candidate_bpm = _read_midi_tempo(midi_path)
            cand_notes = decode_candidate_midi(midi_path)
        except Exception as exc:
            LOGGER.info(
                "retime_decode_failed",
                extra={"candidate_idx": idx, "midi_path": str(midi_path), "error": str(exc)},
            )
            continue

        result = retime_to_primer_density(
            cand_notes,
            primer_notes,
            candidate_bpm=candidate_bpm,
            primer_bpm=primer_bpm,
        )
        if abs(result.stretch_factor - 1.0) > 1e-6:
            try:
                _rewrite_midi_at(midi_path, result.notes)
            except Exception as exc:
                LOGGER.info(
                    "retime_midi_rewrite_failed",
                    extra={"candidate_idx": idx, "midi_path": str(midi_path), "error": str(exc)},
                )
                continue

        candidate["retime"] = {
            "stretch_factor": float(result.stretch_factor),
            "raw_stretch_factor": float(result.raw_stretch_factor),
            "was_clipped": bool(result.was_clipped),
            "primer_density": float(result.primer_active_density),
            "density_before": float(result.candidate_density_before),
            "density_after": float(result.candidate_density_after),
        }
        telemetry.append({
            "candidate_idx": idx,
            "model_id": candidate.get("model_id"),
            **candidate["retime"],
        })
    return telemetry


def _read_midi_tempo(midi_path: Path) -> float:
    """Read the first tempo from a MIDI file; fall back to 120 BPM."""
    from symusic import Score

    try:
        score = Score(str(midi_path))
        if score.tempos:
            return float(score.tempos[0].qpm)
    except Exception:
        pass
    return 120.0


def _rewrite_midi_at(midi_path: Path, retimed_notes_beats: list[NoteEvent]) -> None:
    """Replace the on-disk MIDI's notes with retimed ones, preserving tempo + TPQ."""
    from symusic import Note, Score, Track

    try:
        score = Score(str(midi_path))
        tpq = int(getattr(score, "ticks_per_quarter", getattr(score, "tpq", 480)) or 480)
    except Exception:
        score = Score(480)
        tpq = 480

    # Drop all existing tracks; rebuild a single melody track in their place.
    score.tracks.clear()
    track = Track(name="retimed", program=0, is_drum=False)
    for n in retimed_notes_beats:
        track.notes.append(Note(
            int(round(float(n.onset) * tpq)),
            max(1, int(round(float(n.duration) * tpq))),
            int(n.pitch),
            int(n.velocity),
        ))
    score.tracks.append(track)
    score.dump_midi(str(midi_path))


def _candidate_to_suggestion(candidate: dict[str, Any]) -> MelodySuggestion:
    midi_path = Path(candidate["midi_path"])
    constraint_text = candidate.get("constraint_explanation", "")
    score_breakdown = dict(candidate.get("score_breakdown", {}))
    retime_info = candidate.get("retime")
    if retime_info:
        score_breakdown.update({
            "retime_stretch_factor": float(retime_info.get("stretch_factor", 1.0)),
            "retime_raw_stretch_factor": float(retime_info.get("raw_stretch_factor", 1.0)),
            "retime_clipped": bool(retime_info.get("was_clipped", False)),
            "retime_primer_density": float(retime_info.get("primer_density", 0.0)),
            "retime_density_before": float(retime_info.get("density_before", 0.0)),
            "retime_density_after": float(retime_info.get("density_after", 0.0)),
        })
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
