# Harmony DQN

The harmony module uses the vendored Wiki-DQN-64 checkpoint as the chord model. The default checkpoint path is:

`storage/models/Wiki-DQN-64/epoch14_reward4.298_mle_loss262.858_beta0.700.pth`

Set `HUMMUSE_DQN_CHECKPOINT` in `.env` or the process environment to override it.

## Generation

Chord generation is autoregressive. Each step builds a 177-dimensional DQN state from:

- a 24-dimensional melody window condition
- a 133-dimensional one-hot current-note encoding
- a 20-dimensional one-hot previous-chord encoding

The selected chord from position `t - 1` is encoded back into the next state's previous-chord slot before position `t` is evaluated.

## NoisyNet Inference

The copied DQN architecture includes NoisyLinear layers. During normal inference, `load_model()` calls `model.eval()`, and the local NoisyLinear implementation uses only deterministic `mu` weights and biases in eval mode.

## Emotion Modulation

Emotion modulation is an additive Q-bias on the pitch-class head. `emotion_q_bias(q_pc, valence, arousal, key)` combines a valence bias toward tonic major/minor thirds with an independent arousal tension/release bias. The downstream decoder takes argmax from the biased Q tensor.

Each `ChordDistribution` records the per-pitch-class Q delta and a one-line rationale only when a non-zero bias was actually applied.

Progression-level scoring keeps raw model confidence separate from mood pressure. `ChordProgression.score` and `model_confidence` are computed from the model's un-biased pitch-class Q values, while `mood_alignment` summarizes how strongly the selected pitch classes agree with the applied emotion Q-bias.

## Reward Attribution

Reward attribution is computed after each selected chord with pure inference-time functions:

- `harmony_rule`: consonance and melody/chord interval support
- `progression_penalty`: repetition plus continuous Lerdahl basic-space distance shaping
- `chord_tone_inclusion`: whether the melody note belongs to the selected chord
- `key_fit_proxy`: lightweight proxy for likely key-relative chord choices

`mutual_info` is currently recorded as `0.0`; loading Mutual_Chord checkpoints is a future extension.

### `progression_penalty` Interpretation

This term combines the original repetition rule with a continuous voice-leading distance:

- Repeating the same chord on a bar downbeat contributes `-1`.
- Repeating the same chord within the bar contributes `+1`.
- Moving to a different chord contributes a negative Lerdahl basic-space distance, normalized to stay near the old superstrong-rule range.

The distance is computed by embedding both pitch-class sets into a simple key-relative basic space and counting the symmetric difference. Nearby voice-leading moves therefore receive a smaller negative value than distant chromatic jumps, instead of the old binary "shares any common tone" check.
