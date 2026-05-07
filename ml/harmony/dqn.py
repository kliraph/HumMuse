"""Wrapper for the trained RL-Chord DQN multi-head checkpoint."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ml.harmony.annotation import annotate_chord_positions
from ml.harmony.chord_symbols import QUALITY_TEMPLATES, derive_chord_symbol
from ml.harmony.emotion_modulation import emotion_bias_rationale, emotion_bias_vector, emotion_values
from ml.harmony.explain import build_explanation
from ml.harmony.reward_attribution import compute_attribution
from shared.schemas import ChordDistribution, ChordProgression, EmotionVector

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is present in the app env.
    load_dotenv = None

try:  # Keep importing this module cheap when torch is unavailable.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only on non-ML installs.
    torch = None
    nn = None
    F = None

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DUR2TYPE = {
    4: 0,
    6: 1,
    8: 2,
    12: 3,
    16: 4,
    18: 5,
    24: 6,
    32: 7,
    36: 8,
    48: 9,
    72: 10,
    96: 11,
}
TICKS_PER_QUARTER = 24
BAR_TICKS_4_4 = 96
CONDITION_WINDOW = 8
INPUT_SIZE = 64
HIDDEN_SIZE = 512
CHECKPOINT_FILENAME = "epoch14_reward4.298_mle_loss262.858_beta0.700.pth"
if load_dotenv is not None:
    load_dotenv()
_PACKAGED_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "storage"
    / "models"
    / "Wiki-DQN-64"
    / CHECKPOINT_FILENAME
)
DEFAULT_CHECKPOINT = Path(
    os.getenv(
        "HUMMUSE_DQN_CHECKPOINT",
        str(_PACKAGED_CHECKPOINT),
    )
)

_MODEL_CACHE: "_CachedModel | None" = None


@dataclass(frozen=True)
class _ChordCandidate:
    score: float
    pc_score: float
    octave_score: float
    inversion_score: float
    quality_order: int
    root_pc: int
    octave: int
    inversion: int
    pcs: tuple[int, ...]


if nn is not None:

    class NoisyLinear(nn.Module):
        """Local copy of RL-Chord's factorized noisy linear layer."""

        def __init__(self, in_features: int, out_features: int, std_init: float = 0.2) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init
            self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
            self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
            self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
            self.reset_parameters()
            self.reset_noise()

        def forward(self, x: Any) -> Any:
            if self.training:
                weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
                bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)

        def reset_parameters(self) -> None:
            mu_range = 1 / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

        def reset_noise(self) -> None:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(self._scale_noise(self.out_features))

        def _scale_noise(self, size: int) -> Any:
            x = torch.randn(size)
            return x.sign().mul(x.abs().sqrt())


    class _DQNChord(nn.Module):
        """Local copy of RL-Chord's DQN_Chord architecture with dueling output access."""

        def __init__(
            self,
            condition_window: int,
            input_size: int,
            hidden_size: int,
            n_layers: int = 2,
        ) -> None:
            super().__init__()
            self.condition_window = condition_window
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            self.p_embedding = nn.Embedding(49, 6)
            self.d_embedding = nn.Embedding(12, 4)
            self.b_embedding = nn.Embedding(72, 8)
            self.fc_embedding = nn.Linear(18 * condition_window, 16)
            self.fc_t = nn.Linear(133, 16)
            self.input_linear = nn.Linear(124, 64)
            self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
            self.noisy_value11 = NoisyLinear(hidden_size, 128)
            self.noisy_value12 = NoisyLinear(128, 1)
            self.noisy_advantage11 = NoisyLinear(hidden_size, 128)
            self.noisy_advantage12 = NoisyLinear(128, 1)
            self.sigmoid1 = nn.Sigmoid()
            self.noisy_value21 = NoisyLinear(hidden_size, 128)
            self.noisy_value22 = NoisyLinear(128, 1)
            self.noisy_advantage21 = NoisyLinear(hidden_size, 128)
            self.noisy_advantage22 = NoisyLinear(128, 2)
            self.softmax2 = nn.LogSoftmax(dim=-1)
            self.noisy_value41 = NoisyLinear(hidden_size, 128)
            self.noisy_value42 = NoisyLinear(128, 1)
            self.noisy_advantage41 = NoisyLinear(hidden_size, 128)
            self.noisy_advantage42 = NoisyLinear(128, 4)
            self.softmax4 = nn.LogSoftmax(dim=-1)
            self.noisy_value131 = NoisyLinear(hidden_size, 128)
            self.noisy_value132 = NoisyLinear(128, 1)
            self.noisy_advantage131 = NoisyLinear(hidden_size, 128)
            self.noisy_advantage132 = NoisyLinear(128, 13)
            self.softmax13 = nn.LogSoftmax(dim=-1)

        def forward(self, state: Any, hidden: Any = None) -> tuple[Any, Any, Any, Any, Any]:
            q1, q2, q4, q13, hidden, _ = self.forward_with_dueling(state, hidden)
            return q1, q2, q4, q13, hidden

        def forward_with_dueling(self, state: Any, hidden: Any = None) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
            condition_t, note_t, chord_t_1 = torch.split(state, [24, 133, 20], dim=-1)
            condition_t_pitch, condition_t_duration, condition_t_position = torch.split(condition_t, [8, 8, 8], dim=-1)
            condition_t_pitch = condition_t_pitch.long()
            condition_t_duration = condition_t_duration.long()
            condition_t_position = condition_t_position.long()
            note_t_pitch, note_t_duration, note_t_position = torch.split(note_t, [49, 12, 72], dim=-1)

            p_embedding = self.p_embedding(condition_t_pitch)
            d_embedding = self.d_embedding(condition_t_duration)
            b_embedding = self.b_embedding(condition_t_position)
            embeddings = torch.cat((p_embedding, d_embedding, b_embedding), -1)
            embeddings = embeddings.view(p_embedding.shape[0], p_embedding.shape[1], -1)
            embeddings = self.fc_embedding(embeddings)
            note_t = torch.cat((note_t_pitch, note_t_duration, note_t_position), -1)
            note_t = self.fc_t(note_t)
            condition = torch.cat((embeddings, note_t), -1)
            inputs = torch.cat((chord_t_1, note_t_position), -1)
            inputs = torch.cat((inputs, condition), -1)
            inputs = self.input_linear(inputs)
            inputs, hidden = self.lstm(inputs, hidden)

            value1, advantage1, q1 = self._dueling_head(
                inputs,
                self.noisy_value11,
                self.noisy_value12,
                self.noisy_advantage11,
                self.noisy_advantage12,
            )
            value2, advantage2, q2 = self._dueling_head(
                inputs,
                self.noisy_value21,
                self.noisy_value22,
                self.noisy_advantage21,
                self.noisy_advantage22,
            )
            value4, advantage4, q4 = self._dueling_head(
                inputs,
                self.noisy_value41,
                self.noisy_value42,
                self.noisy_advantage41,
                self.noisy_advantage42,
            )
            value13, advantage13, q13 = self._dueling_head(
                inputs,
                self.noisy_value131,
                self.noisy_value132,
                self.noisy_advantage131,
                self.noisy_advantage132,
            )
            dueling = {
                "value_rest": value1,
                "value_octave": value2,
                "value_inversion": value4,
                "value_pc": value13,
                "advantage_rest": advantage1,
                "advantage_octave": advantage2,
                "advantage_inversion": advantage4,
                "advantage_pc": advantage13,
            }
            return q1, q2, q4, q13, hidden, dueling

        def reset_noise(self) -> None:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

        def act(self, state: Any, hidden: Any = None) -> tuple[Any, Any, Any, Any, Any]:
            q1, q2, q4, q13, hidden = self.forward(state, hidden)
            return self.sigmoid1(q1), self.softmax2(q2), self.softmax4(q4), self.softmax13(q13), hidden

        @staticmethod
        def _dueling_head(inputs: Any, value_1: Any, value_2: Any, advantage_1: Any, advantage_2: Any) -> tuple[Any, Any, Any]:
            advantage = advantage_2(F.relu(advantage_1(inputs)))
            value = value_2(F.relu(value_1(inputs)))
            q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return value, advantage, q_value


@dataclass(frozen=True)
class _CachedModel:
    checkpoint_path: Path
    model: Any
    checkpoint_metadata: dict[str, Any]


def generate_chords(
    melody_notes: list[Any],
    key: str | None,
    emotion_vector: EmotionVector | dict[str, float] | None,
    top_k: int = 3,
    *,
    checkpoint_path: str | Path | None = None,
    tempo_bpm: float = 120.0,
    input_unit: Literal["beats", "seconds"] = "beats",
) -> list[ChordProgression]:
    """Generate ranked DQN chord progressions for a melody.

    Top-k output uses independent deterministic rollouts. Each step ranks
    root/quality chord templates by joint Q across pitch-class, inversion, and
    octave heads, then takes the K-th candidate for that rollout. NoisyLinear
    layers are disabled for inference so repeated API calls are stable.
    """

    if top_k <= 0:
        return []

    events = notes_to_events(melody_notes, tempo_bpm=tempo_bpm, input_unit=input_unit)
    if not events:
        return []

    cached = load_model(checkpoint_path)
    progressions = []
    for rank in range(top_k):
        rollout = _rollout(cached.model, events, key=key, emotion_vector=emotion_vector, candidate_rank=rank)
        model_confidence = _progression_model_confidence(rollout["distributions"])
        mood_alignment = _progression_mood_alignment(rollout["distributions"])
        progressions.append(
            ChordProgression(
                chords=rollout["chords"],
                score=model_confidence,
                model_confidence=model_confidence,
                mood_alignment=mood_alignment,
                harmonic_function=None,
                explanation=_progression_explanation(
                    key,
                    emotion_vector,
                    rank,
                    cached.checkpoint_metadata,
                    rollout["distributions"],
                    rollout["derivations"],
                    model_confidence=model_confidence,
                    mood_alignment=mood_alignment,
                ),
                native_distributions=rollout["distributions"],
                chord_annotations=rollout["annotations"],
                chord_explanations=rollout["explanations"],
            )
        )
    return progressions


def load_model(checkpoint_path: str | Path | None = None) -> _CachedModel:
    """Load and cache the trained DQN checkpoint selected by path or .env."""

    global _MODEL_CACHE
    _require_torch()
    resolved_path = Path(checkpoint_path or _default_checkpoint())
    if _MODEL_CACHE is not None and _MODEL_CACHE.checkpoint_path == resolved_path:
        return _MODEL_CACHE

    if not resolved_path.exists():
        raise FileNotFoundError(f"DQN checkpoint not found: {resolved_path}")

    model = _DQNChord(condition_window=CONDITION_WINDOW, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    checkpoint = _torch_load(resolved_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "test_reward": _last_json_float(checkpoint.get("test_rewards")),
        "test_count": _last_json_float(checkpoint.get("test_nums")),
        "checkpoint_path": str(resolved_path),
    }
    _MODEL_CACHE = _CachedModel(checkpoint_path=resolved_path, model=model, checkpoint_metadata=metadata)
    return _MODEL_CACHE


def clear_model_cache() -> None:
    """Clear the module-level model cache for tests or checkpoint swaps."""

    global _MODEL_CACHE
    _MODEL_CACHE = None


def notes_to_events(
    melody_notes: list[Any],
    *,
    tempo_bpm: float = 120.0,
    input_unit: Literal["beats", "seconds"] = "beats",
    bar_ticks: int = BAR_TICKS_4_4,
) -> list[tuple[int, int, int]]:
    """Convert HumMuse/Basic-Pitch style notes to RL-Chord event triples."""

    normalized_notes = sorted((_normalize_note(note) for note in melody_notes), key=lambda note: note["onset"])
    events: list[tuple[int, int, int]] = []
    cursor = 0
    for note in normalized_notes:
        onset_ticks = _time_to_ticks(note["onset"], tempo_bpm=tempo_bpm, input_unit=input_unit)
        if onset_ticks > cursor:
            rest_ticks = duration_revise(onset_ticks - cursor)
            if rest_ticks in DUR2TYPE:
                events.append((0, DUR2TYPE[rest_ticks], (cursor % bar_ticks) // 2))
            cursor = onset_ticks

        duration_ticks = duration_revise(_time_to_ticks(note["duration"], tempo_bpm=tempo_bpm, input_unit=input_unit))
        if duration_ticks not in DUR2TYPE:
            continue

        events.append((note["pitch"], DUR2TYPE[duration_ticks], (cursor % bar_ticks) // 2))
        cursor += duration_ticks
    return events


def build_dqn_state(
    events: list[tuple[int, int, int]],
    prev_chord_one_hot: list[float] | Any,
    *,
    position: int = 0,
    window: int = CONDITION_WINDOW,
) -> Any:
    """Build one 1 x 1 x 177 DQN state: 24 condition + 133 note + 20 previous chord."""

    _require_torch()
    condition = _condition_window(events, position=position, window=window)
    note = _note_one_hot(events[position] if position < len(events) else (0, 0, 0))
    previous = torch.as_tensor(prev_chord_one_hot, dtype=torch.float32).view(20)
    state = torch.cat((condition.float(), note, previous), dim=-1)
    return state.view(1, 1, -1)


def duration_revise(duration_ticks: int) -> int:
    """Match RL-Chord's duration quantization preference."""

    if duration_ticks % 4 == 0 or duration_ticks % 6 == 0:
        return duration_ticks
    next_multiple_4 = duration_ticks + (4 - duration_ticks % 4)
    next_multiple_6 = duration_ticks + (6 - duration_ticks % 6)
    return next_multiple_4 if next_multiple_4 < next_multiple_6 else next_multiple_6


def encode_chord_to_20dim(selected: dict[str, Any] | Any) -> list[float]:
    """Encode selected multi-head chord actions into RL-Chord's 20-dim previous-chord vector."""

    if hasattr(selected, "model_dump"):
        selected = selected.model_dump()
    vector = [0.0] * 20
    if selected.get("is_rest"):
        vector[0] = 1.0
        return vector

    octave_index = int(selected.get("octave", 2)) - 2
    inversion = int(selected.get("inversion", 0))
    pcs = [int(pc) % 12 for pc in selected.get("pcs", [])]
    vector[1 + max(0, min(1, octave_index))] = 1.0
    vector[3 + max(0, min(3, inversion))] = 1.0
    if len(pcs) == 3:
        vector[19] = 1.0
    for pc in pcs:
        vector[7 + pc] = 1.0
    return vector


def start_chord_one_hot() -> list[float]:
    vector = [0.0] * 20
    vector[0] = 1.0
    return vector


def act_outputs(q1: Any, q2: Any, q4: Any, q13: Any) -> dict[str, Any]:
    """Convert DQN Q heads into policy probabilities for display and decoding."""

    rest_prob = float(torch.sigmoid(q1).view(-1)[0].item())
    octave_probs = torch.softmax(q2, dim=-1).view(-1)
    inversion_probs = torch.softmax(q4, dim=-1).view(-1)
    pitch_probs = torch.softmax(q13, dim=-1).view(-1)
    membership = [float(pitch_probs[index].item()) for index in range(13)]
    return {
        "rest": {"rest": rest_prob, "chord": 1.0 - rest_prob},
        "octave": {"2": float(octave_probs[0].item()), "3": float(octave_probs[1].item())},
        "inversion": {str(index): float(inversion_probs[index].item()) for index in range(4)},
        "pitch_class": {PC_NAMES[index]: float(pitch_probs[index].item()) for index in range(12)},
        "pitch_class_membership": membership,
        "pitch_class_labels": [*PC_NAMES, "triad_sentinel"],
    }


def _rollout(
    model: Any,
    events: list[tuple[int, int, int]],
    *,
    key: str | None,
    emotion_vector: EmotionVector | dict[str, float] | None,
    candidate_rank: int,
) -> dict[str, Any]:
    hidden = None
    prev_one_hot = start_chord_one_hot()
    prev_chord_pcs: list[int] = []
    prev_chord_symbol: str | None = None
    prev_selected: dict[str, Any] | None = None
    distributions: list[ChordDistribution] = []
    derivations = []

    with torch.no_grad():
        for position, event in enumerate(events):
            state = build_dqn_state(events, prev_one_hot, position=position)
            q1, q2, q4, q13, hidden, dueling = model.forward_with_dueling(state, hidden)
            q13_biased, emotion_bias = _apply_emotion_q_bias(q13, emotion_vector, key=key)
            policy = act_outputs(q1, q2, q4, q13_biased)
            selected = decode(policy, q1, q2, q4, q13_biased, candidate_rank=candidate_rank)
            reward_attribution = compute_attribution(selected, event, prev_selected, key)
            distribution = build_chord_distribution(
                position,
                q1,
                q2,
                q4,
                q13_biased,
                policy,
                selected,
                dueling,
                event,
                emotion_bias=emotion_bias,
                reward_attribution=reward_attribution,
                prev_chord_pcs=prev_chord_pcs,
                prev_chord_symbol=prev_chord_symbol,
            )
            derivation = derive_chord_symbol(
                distribution,
                key=key,
                previous_symbol=prev_chord_symbol,
                previous_pcs=prev_chord_pcs,
            )
            distributions.append(distribution)
            derivations.append(derivation)
            prev_one_hot = encode_chord_to_20dim(distribution.selected)
            prev_chord_pcs = list(derivation.pitch_classes)
            prev_chord_symbol = derivation.symbol
            prev_selected = distribution.selected.model_dump()

    annotations = annotate_chord_positions(distributions, events, key=key, derivations=derivations)
    explanations = [
        build_explanation(distribution, derivation)
        for distribution, derivation in zip(distributions, derivations)
    ]
    return {
        "chords": [derivation.symbol for derivation in derivations],
        "distributions": distributions,
        "derivations": derivations,
        "annotations": annotations,
        "explanations": explanations,
    }


def decode(policy: dict[str, Any], q1: Any, q2: Any, q4: Any, q13: Any, *, candidate_rank: int = 0) -> dict[str, Any]:
    """Decode one DQN decision from the ranked joint-Q chord-template list."""

    rest_probability = float(policy["rest"]["rest"])
    if rest_probability > 0.5 and candidate_rank == 0:
        return {"is_rest": True, "octave": None, "inversion": None, "pcs": [], "pc_names": []}

    candidates = _rank_chord_candidates(_flat(q2), _flat(q4), _flat(q13))
    if not candidates:
        pc_ranked = _ranked_indices(_flat(q13)[:12])
        selected_pcs = tuple(sorted(pc_ranked[:3]))
        octave = _ranked_indices(_flat(q2))[0] + 2
        inversion = _ranked_indices(_flat(q4))[0]
    else:
        adjusted_rank = candidate_rank - 1 if rest_probability > 0.5 else candidate_rank
        candidate = candidates[min(max(0, adjusted_rank), len(candidates) - 1)]
        selected_pcs = candidate.pcs
        octave = candidate.octave
        inversion = candidate.inversion
    return {
        "is_rest": False,
        "octave": octave,
        "inversion": inversion,
        "pcs": list(selected_pcs),
        "pc_names": [PC_NAMES[pitch_class] for pitch_class in selected_pcs],
    }


def _rank_chord_candidates(q2_values: list[float], q4_values: list[float], q13_values: list[float]) -> list[_ChordCandidate]:
    """Rank root/quality/octave/inversion candidates by summed DQN head Q."""

    if len(q2_values) < 2 or len(q4_values) < 4 or len(q13_values) < 13:
        return []

    candidates: list[_ChordCandidate] = []
    for octave_index, octave_score in enumerate(q2_values[:2]):
        octave = octave_index + 2
        for inversion, inversion_score in enumerate(q4_values[:4]):
            for root_pc in range(12):
                for quality_order, template in enumerate(QUALITY_TEMPLATES):
                    if inversion >= len(template.intervals):
                        continue
                    pcs = tuple(sorted({(root_pc + interval) % 12 for interval in template.intervals}))
                    if len(pcs) != len(template.intervals):
                        continue
                    pc_score = sum(q13_values[pitch_class] for pitch_class in pcs)
                    if len(pcs) == 3:
                        pc_score += q13_values[12]
                    score = pc_score + float(octave_score) + float(inversion_score)
                    candidates.append(
                        _ChordCandidate(
                            score=float(score),
                            pc_score=float(pc_score),
                            octave_score=float(octave_score),
                            inversion_score=float(inversion_score),
                            quality_order=quality_order,
                            root_pc=root_pc,
                            octave=octave,
                            inversion=inversion,
                            pcs=pcs,
                        )
                    )

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            -candidate.pc_score,
            -candidate.octave_score,
            -candidate.inversion_score,
            candidate.quality_order,
            candidate.root_pc,
            candidate.octave,
            candidate.inversion,
        ),
    )


def build_chord_distribution(
    position: int,
    q1: Any,
    q2: Any,
    q4: Any,
    q13: Any,
    policy: dict[str, Any],
    selected: dict[str, Any],
    dueling: dict[str, Any],
    event: tuple[int, int, int],
    *,
    emotion_bias: dict[str, Any] | None,
    reward_attribution: dict[str, float] | None,
    prev_chord_pcs: list[int],
    prev_chord_symbol: str | None,
) -> ChordDistribution:
    q_rest = _flat(q1)[0]
    q_octave = _flat(q2)
    q_inversion = _flat(q4)
    q_pitch_class = _flat(q13)
    value_rest = _flat(dueling["value_rest"])[0]
    value_octave = _flat(dueling["value_octave"])[0]
    value_inversion = _flat(dueling["value_inversion"])[0]
    value_pc = _flat(dueling["value_pc"])[0]
    pitch, duration, event_position = event
    return ChordDistribution(
        position=position,
        policy=policy,
        q_values={
            "rest": q_rest,
            "octave": q_octave,
            "inversion": q_inversion,
            "pitch_class": q_pitch_class,
        },
        q_margin={
            "rest": _top_margin([q_rest, 0.0]),
            "octave": _top_margin(q_octave),
            "inversion": _top_margin(q_inversion),
            "pitch_class": _top_margin(q_pitch_class),
        },
        dueling={
            "value_rest": value_rest,
            "value_octave": value_octave,
            "value_inversion": value_inversion,
            "value_pc": value_pc,
            "advantage_rest": _flat(dueling["advantage_rest"]),
            "advantage_octave": _flat(dueling["advantage_octave"]),
            "advantage_inversion": _flat(dueling["advantage_inversion"]),
            "advantage_pc": _flat(dueling["advantage_pc"]),
        },
        context={
            "prev_chord_pcs": prev_chord_pcs,
            "prev_chord_symbol": prev_chord_symbol,
            "current_note": {"pitch": pitch, "duration": duration, "position": event_position},
        },
        reward_attribution=reward_attribution,
        emotion_bias=emotion_bias,
        selected=selected,
        model_kind="dqn",
        noise_state="disabled",
        noise_samples=None,
    )


def _apply_emotion_q_bias(
    q13: Any,
    emotion_vector: EmotionVector | dict[str, float] | None,
    *,
    key: str | None,
) -> tuple[Any, dict[str, Any] | None]:
    values = emotion_values(emotion_vector)
    if values is None:
        return q13, None
    valence, arousal = values
    bias = emotion_bias_vector(q13, valence=valence, arousal=arousal, key=key)
    flat = _flat(bias)
    if not any(abs(value) > 1e-12 for value in flat):
        return q13, None
    return q13 + bias, {
        "applied": True,
        "q_delta_per_pc": flat,
        "rationale": emotion_bias_rationale(valence, arousal, key=key),
    }


def _condition_window(events: list[tuple[int, int, int]], *, position: int, window: int) -> Any:
    total = len(events)
    if position - window / 2 >= 0 and position + window / 2 - 1 < total:
        window_start, window_end = int(position - window / 2), int(position + window / 2)
    elif position - window / 2 < 0:
        window_start, window_end = 0, min(window, total)
    else:
        window_start, window_end = max(0, total - window), total

    event_window = list(events[window_start:window_end])
    while len(event_window) < window:
        event_window.append((0, 0, 0))

    condition = torch.zeros(24, dtype=torch.long)
    for offset, (pitch, duration_type, bar_position) in enumerate(event_window[:window]):
        condition[offset] = _pitch_index(pitch)
        condition[offset + 8] = min(11, max(0, int(duration_type)))
        condition[offset + 16] = min(71, max(0, int(bar_position)))
    return condition


def _note_one_hot(event: tuple[int, int, int]) -> Any:
    pitch, duration_type, bar_position = event
    note = torch.zeros(133, dtype=torch.float32)
    note[_pitch_index(pitch)] = 1.0
    note[49 + min(11, max(0, int(duration_type)))] = 1.0
    note[49 + 12 + min(71, max(0, int(bar_position)))] = 1.0
    return note


def _pitch_index(pitch: int) -> int:
    return 0 if pitch == 0 else max(1, min(48, int(pitch) - 47))


def _progression_model_confidence(distributions: list[ChordDistribution]) -> float:
    if not distributions:
        return 0.0
    confidences = []
    for distribution in distributions:
        selected = distribution.selected
        if selected["is_rest"]:
            confidences.append(distribution.rest["rest"])
            continue
        octave = distribution.octave.get(str(selected["octave"]), 0.0)
        inversion = distribution.inversion.get(str(selected["inversion"]), 0.0)
        raw_pc_probs = _model_pitch_class_probs(distribution)
        pitch = sum(raw_pc_probs[pc] for pc in selected["pcs"]) / max(1, len(selected["pcs"]))
        confidences.append((octave + inversion + pitch) / 3)
    base_score = sum(confidences) / len(confidences)
    return max(0.0, min(1.0, round(base_score, 4)))


def _progression_mood_alignment(distributions: list[ChordDistribution]) -> float | None:
    alignments = []
    for distribution in distributions:
        bias = distribution.emotion_bias
        if bias is None or not bias.applied:
            continue
        deltas = list(bias.q_delta_per_pc)
        scale = max((abs(delta) for delta in deltas[:12]), default=0.0)
        if distribution.selected["is_rest"] or scale <= 1e-12:
            alignments.append(0.5)
            continue
        selected_deltas = [
            deltas[pc]
            for pc in distribution.selected["pcs"]
            if 0 <= pc < len(deltas)
        ]
        if not selected_deltas:
            alignments.append(0.5)
            continue
        mean_delta = sum(selected_deltas) / len(selected_deltas)
        alignments.append(max(0.0, min(1.0, (mean_delta / scale + 1.0) / 2.0)))
    if not alignments:
        return None
    return round(sum(alignments) / len(alignments), 4)


def _model_pitch_class_probs(distribution: ChordDistribution) -> list[float]:
    q_values = list(distribution.q_values.pitch_class)
    bias = distribution.emotion_bias
    if bias is not None and bias.q_delta_per_pc:
        q_values = [
            value - bias.q_delta_per_pc[index] if index < len(bias.q_delta_per_pc) else value
            for index, value in enumerate(q_values)
        ]
    return _softmax_values(q_values)


def _softmax_values(values: list[float]) -> list[float]:
    if not values:
        return []
    shifted = [float(value) - max(values) for value in values]
    exps = [math.exp(value) for value in shifted]
    total = sum(exps)
    return [value / total for value in exps] if total else [0.0 for _ in values]


def _progression_explanation(
    key: str | None,
    emotion_vector: EmotionVector | dict[str, float] | None,
    rank: int,
    metadata: dict[str, Any],
    distributions: list[ChordDistribution],
    derivations: list[Any],
    *,
    model_confidence: float,
    mood_alignment: float | None,
) -> str:
    key_text = key or "unknown key"
    emotion_text = "without emotion modulation"
    if emotion_vector is not None:
        emotion_text = "with emotion-biased pitch-class decoding"
    epoch = metadata.get("epoch")
    checkpoint_name = Path(str(metadata.get("checkpoint_path", "DQN checkpoint"))).name
    strategy = "greedy joint-Q template rollout" if rank == 0 else f"joint-Q template candidate rollout {rank + 1}"
    explanations = [
        build_explanation(distribution, derivation)
        for distribution, derivation in zip(distributions, derivations)
    ]
    average_q = _mean([explanation.q_chosen for explanation in explanations])
    weakest = min(explanations, key=lambda explanation: explanation.margin, default=None)
    reward_totals = _reward_totals(distributions)
    reward_text = ", ".join(
        f"{name}={value:.2f}"
        for name, value in sorted(reward_totals.items())
    )
    weakest_text = "none"
    if weakest is not None:
        weakest_text = f"position {weakest.position + 1} {weakest.chord_symbol} margin {weakest.margin:.2f}"
    mood_text = "n/a" if mood_alignment is None else f"{mood_alignment:.2f}"
    return (
        f"DQN candidate {rank + 1} for {key_text}, generated with {strategy} from cached "
        f"{checkpoint_name} at epoch {epoch}; average chosen Q={average_q:.2f}, weakest {weakest_text}; "
        f"model confidence={model_confidence:.2f}, mood alignment={mood_text}; "
        f"reward totals: {reward_text}; {emotion_text}."
    )


def _reward_totals(distributions: list[ChordDistribution]) -> dict[str, float]:
    totals = {
        "harmony_rule": 0.0,
        "progression_penalty": 0.0,
        "chord_tone_inclusion": 0.0,
        "mutual_info": 0.0,
        "key_fit_proxy": 0.0,
    }
    for distribution in distributions:
        if distribution.reward_attribution is None:
            continue
        totals["harmony_rule"] += distribution.reward_attribution.harmony_rule
        totals["progression_penalty"] += distribution.reward_attribution.progression_penalty
        totals["chord_tone_inclusion"] += distribution.reward_attribution.chord_tone_inclusion
        totals["mutual_info"] += distribution.reward_attribution.mutual_info
        totals["key_fit_proxy"] += distribution.reward_attribution.key_fit_proxy
    return {key: round(value, 6) for key, value in totals.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_checkpoint() -> Path:
    if load_dotenv is not None:
        load_dotenv()
    if "HUMMUSE_DQN_CHECKPOINT" in os.environ:
        return Path(os.environ["HUMMUSE_DQN_CHECKPOINT"])
    return _PACKAGED_CHECKPOINT


def _normalize_note(note: Any) -> dict[str, float | int]:
    if hasattr(note, "model_dump"):
        note = note.model_dump()
    elif not isinstance(note, dict):
        note = vars(note)

    pitch = note.get("pitch")
    if isinstance(pitch, str):
        pitch = _pitch_to_midi(pitch)

    onset = note.get("onset", note.get("start_beat", note.get("start", 0.0)))
    duration = note.get("duration", note.get("duration_beats", 0.0))
    return {
        "pitch": int(pitch),
        "onset": float(onset),
        "duration": float(duration),
    }


def _time_to_ticks(value: float, *, tempo_bpm: float, input_unit: Literal["beats", "seconds"]) -> int:
    if input_unit == "seconds":
        return round(value * (tempo_bpm / 60.0) * TICKS_PER_QUARTER)
    return round(value * TICKS_PER_QUARTER)


def _pitch_to_midi(pitch: str) -> int:
    pitch_classes = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    note_name = pitch[:-1]
    octave = int(pitch[-1])
    return (octave + 1) * 12 + pitch_classes[note_name]


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch.
        return torch.load(path, map_location="cpu")


def _last_json_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return None if not value else float(value[-1])
    return float(value)


def _flat(tensor: Any) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().view(-1).tolist()]


def _top_margin(values: list[float]) -> float:
    if not values:
        return 0.0
    ranked = sorted(values, reverse=True)
    if len(ranked) == 1:
        return ranked[0]
    return ranked[0] - ranked[1]


def _ranked_indices(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: values[index], reverse=True)


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("The DQN harmony wrapper requires torch to be installed.")
