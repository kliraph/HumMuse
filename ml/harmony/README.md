# Harmony DQN

The harmony module uses the vendored Wiki-DQN-64 checkpoint as the chord model. The default checkpoint path is:

`storage/models/Wiki-DQN-64/epoch24_reward4.342_mle_loss260.997_beta0.700.pth`

Set `HUMMUSE_DQN_CHECKPOINT` in `.env` or the process environment to override it.

## Generation

Chord generation is autoregressive. Each step builds a 177-dimensional DQN state from:

- a 24-dimensional melody window condition
- a 133-dimensional one-hot current-note encoding
- a 20-dimensional one-hot previous-chord encoding

The selected chord from position `t - 1` is encoded back into the next state's previous-chord slot before position `t` is evaluated.

## Harmonizer-Input Quantization

The DQN was trained on metrically-quantized lead-sheet data, so it relies on a clean `bar_position` / `duration_type` grid to decide *where* chords change. Hummed input does not provide this: PESTO produces 10 ms-resolution onsets that carry the natural rubato of humming, and once converted to beats those onsets scatter across many distinct bar positions with no repeating grid. The model has no downbeat structure to anchor to and collapses to a single sustained chord (typically the tonic, or an off-key drift).

`ml.harmony.quantize.notes_to_harmony_grid` snaps a **throwaway copy** of the melody onto a quarter-note (1.0-beat) grid before chord inference. This is the only thing the chord model sees; the melody artifact the user keeps stays unquantized.

This is deliberately a different decision from the melody-extraction pipeline, which **dropped** quantization (see the note in `ml/melody_sketchpad/pipeline.py`). Snapping the *playable* melody crushes hum rubato and creates phantom polyphony at MIDI export — bad for the melody the user hears. Snapping a *throwaway harmonizer-input copy* has neither problem: it is never exported, played back, or shown. The two uses do not conflict.

Notes:

- The grid is **quarter-note by default** (`grid_beats=1.0`). Coarse is intentional — slow hums quantize more cleanly at quarter resolution than at eighth, which preserves too much timing irregularity.
- Durations below `min_duration_beats` are clamped up so every note keeps a positive duration and contributes one chord position.
- Quantization quality depends on the tempo used for the upstream seconds→beats conversion. When librosa beat tracking fails on a short hum (it returns a flat 100 BPM safety net), the melody pipeline now falls back to a **median inter-onset-interval tempo estimate** (`estimate_tempo_from_onsets`, octave-folded into 60–160 BPM, recorded as `tempo_source="ioi_fallback"`) instead of the flat default. A user-provided or successfully-detected tempo always wins.

Example recovery on a real C-minor hum (`pesto_hum_render.wav`): the unquantized greedy candidate collapses to `G♯/C × 17` (off-key); the quantized greedy candidate recovers to `Cm → Fm7 → G♯6 → A♯7` (i → iv7 → ♭VI6 → ♭VII7), one chord change per bar.

**Caveat:** this is currently validated on a single melody. The honest next step is a full hum-set comparison through `evaluation/system_eval.py` on a harmony-quality metric (chord variety / key-adherence), quantized vs unquantized. The prior decision to drop quantization was validated against *melody* metrics, not *harmony* metrics, so the harmonizer-input use is genuinely unmeasured at set scale.

## NoisyNet Inference

The copied DQN architecture includes NoisyLinear layers. During normal inference, `load_model()` calls `model.eval()`, and the local NoisyLinear implementation uses only deterministic `mu` weights and biases in eval mode.

## Emotion Modulation

Emotion modulation is an additive Q-bias on the pitch-class head. `emotion_q_bias(q_pc, valence, arousal, key)` combines a mode-aware valence bias with an independent Lerdahl-based arousal tension/release bias. The downstream decoder takes argmax from the biased Q tensor.

The valence bias depends on the **declared mode** of the key, not just the sign of valence:

| Declared mode | Valence sign | Boost                                                      | Suppress                          |
|---------------|--------------|------------------------------------------------------------|-----------------------------------|
| major         | positive     | major 3rd ({4})                                            | minor 3rd ({3})                   |
| major         | negative     | minor 3rd ({3}) (modal mixture)                            | major 3rd ({4})                   |
| minor         | positive     | tonic minor triad + raised 6 + raised 7 (Dorian / harmonic minor brightening) | — (♭3 is **never** suppressed)    |
| minor         | negative     | tonic minor triad + ♭6 (Phrygian / Aeolian darkening)      | — (♭3 is **never** suppressed)    |

In a **major** key the major triad and its parallel minor share root and fifth, so the third is the *only* tone that distinguishes brightness from darkness — the bias therefore moves just the third. (An earlier implementation boosted the whole major triad and suppressed the whole minor triad; the shared root+fifth cancelled to zero, so the effective bias was identical, but the recorded `q_delta_per_pc` and rationale now state plainly that only the third moves.)

The minor-key branches never put the tonic minor 3rd in the suppress set. Earlier versions used the same "boost major-triad PCs when valence > 0" rule regardless of mode, which silently demoted ♭3 in declared minor keys and pushed the model toward chord vocabularies that excluded it. The new branches preserve the minor character of the declared key while still expressing brightness/darkness through scale-degree color tones.

(Note: at equal `valence` magnitude the major-key bias moves one pitch class while the minor-key bias moves four or five, so aggregate valence pressure is currently stronger in minor keys than major. This is a known imbalance, left as a tuning question for the evaluation harness rather than the inference code.)

Each `ChordDistribution` records the per-pitch-class Q delta and a one-line rationale only when a non-zero bias was actually applied.

Progression-level scoring keeps raw model confidence separate from mood pressure. `ChordProgression.score` and `model_confidence` are computed from the model's un-biased pitch-class Q values, while `mood_alignment` summarizes how strongly the selected pitch classes agree with the applied emotion Q-bias.

"Mood" here means the **whole** emotion vector — both valence *and* arousal — not just valence. `mood_alignment` is therefore non-null whenever *any* emotion bias was applied, including a pure-arousal request (`valence=0, arousal>0`), which is a real mood (high activation at neutral valence — tense/alert), not a "no mood" case. The metric reports whether the chosen chords leaned toward the tension/release and triad-quality targets that the full vector pushed for.

## Key Constraint (Road B)

The trained DQN does not receive the declared key as an input — it has to infer the tonal centre from the local melody window. For chromatic-leaning or short melodies this is unreliable, and the model can lock onto an off-key chord vocabulary (e.g. C#-rooted chords in a melody declared as C minor). The key constraint module applies an inference-time **soft key filter** to push back against this drift.

For every chord position, after the emotion bias has been applied, every pitch class that is **not** in the declared key's natural scale has `lambda_key` subtracted from its Q-value. The filter:

- is **soft**: chromatic chords (secondary dominants, modal mixture) are still reachable when the model is confident enough to overcome the penalty;
- is **deterministic**: same melody and key always produce the same demotion vector;
- leaves the **triad sentinel slot (PC 12)** untouched;
- is **disabled** when no key is declared or when `lambda_key == 0`.

The default `lambda_key` is `1.5`. Override it with the `HUMMUSE_KEY_FILTER_LAMBDA` environment variable. The scale tables are the natural major and natural minor scales of the declared tonic; harmonic and melodic minor variants are not currently distinguished.

Each `ChordDistribution` records a `key_constraint` payload with the declared key, the in-scale PCs, the demoted out-of-scale PCs, the lambda used, the per-PC Q delta, and a human-readable rationale.

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
