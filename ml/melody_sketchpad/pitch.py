"""Pitch extraction helpers for melody sketchpad.

PESTO (Riou et al., ISMIR 2023) is used as the monophonic f0 backbone. A
median-filtered jump-threshold segmenter then collapses frame-wise pitch
into note events. PESTO replaced Basic Pitch because Basic Pitch is a
polyphonic instrument transcriber and oversegments humming on overtone
fluctuations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module

import numpy as np

from ml.melody_sketchpad.notes import confidence_to_velocity
from shared.schemas import NoteEvent

try:
    import librosa
except ModuleNotFoundError:  # pragma: no cover - optional dependency until installed in the env
    librosa = None

DEFAULT_MINIMUM_FREQUENCY = 80.0
DEFAULT_MAXIMUM_FREQUENCY = 800.0
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 80.0
DEFAULT_PITCHED_RATIO_THRESHOLD = 0.4
DEFAULT_FALLBACK_PITCH = 60
DEFAULT_ONSET_HOP_LENGTH = 512
DEFAULT_RHYTHM_CONFIDENCE_FLOOR = 0.2
DEFAULT_PESTO_STEP_MS = 10.0
DEFAULT_VOICING_THRESHOLD = 0.5
DEFAULT_MAX_JUMP_SEMITONES = 1.0
DEFAULT_ONSET_DELTA = 0.07
# Hop length (samples) used by both PESTO (via DEFAULT_PESTO_STEP_MS) and the
# librosa onset detector, so onset frame indices align 1:1 with PESTO frames.
DEFAULT_ONSET_HOP_SAMPLES_AT_16K = 160  # 10 ms at 16 kHz
# Onset-split gating thresholds (see _onset_split_passes_filter). Tuned so
# release transients, passing tones, and overtone spikes are rejected while
# genuine repeated notes in legato singing are honoured.
DEFAULT_ONSET_FILTER_MAX_PITCH_STDDEV_SEMITONES = 0.5
DEFAULT_ONSET_FILTER_AMPLITUDE_VALLEY_RATIO = 0.85
DEFAULT_RMS_FRAME_SAMPLES = 400  # 25 ms at 16 kHz

_pesto_predict = None


def seed_from_size(size: int) -> int:
    return max(1, min(size // 1000, 4))


def _ensure_pesto_loaded() -> None:
    global _pesto_predict
    if _pesto_predict is not None:
        return
    pesto_module = import_module("pesto")
    _pesto_predict = pesto_module.predict


def _pesto_extract_frames(
    audio: np.ndarray, sample_rate: int, *, step_ms: float = DEFAULT_PESTO_STEP_MS
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run PESTO and return (pitch_hz, confidence, hop_seconds)."""
    _ensure_pesto_loaded()
    import torch  # imported lazily so tests that monkeypatch this helper don't pay the cost

    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    audio_tensor = torch.from_numpy(waveform)
    _, pitch_hz, confidence, _ = _pesto_predict(audio_tensor, sample_rate, step_size=float(step_ms))
    return pitch_hz.detach().cpu().numpy(), confidence.detach().cpu().numpy(), float(step_ms) / 1000.0


@dataclass(frozen=True)
class PitchConfidenceResult:
    notes: list[NoteEvent]
    pitched_ratio: float
    should_trigger_fallback: bool
    used_fallback: bool = False
    pitched_notes: list[NoteEvent] = field(default_factory=list)


def calculate_pitched_ratio(note_events: list[NoteEvent], audio_duration: float) -> float:
    """Return the proportion of the clip covered by pitched note events."""
    if audio_duration <= 0 or not note_events:
        return 0.0

    intervals = sorted(
        (
            max(0.0, float(note.onset)),
            max(0.0, min(audio_duration, float(note.onset) + float(note.duration))),
        )
        for note in note_events
        if float(note.duration) > 0
    )
    if not intervals:
        return 0.0

    merged_duration = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if end <= current_start:
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged_duration += max(0.0, current_end - current_start)
        current_start, current_end = start, end

    merged_duration += max(0.0, current_end - current_start)
    return max(0.0, min(1.0, merged_duration / audio_duration))


def analyse_pitch_confidence(
    note_events: list[NoteEvent],
    audio_duration: float,
    *,
    pitched_ratio_threshold: float = DEFAULT_PITCHED_RATIO_THRESHOLD,
) -> PitchConfidenceResult:
    pitched_ratio = calculate_pitched_ratio(note_events, audio_duration)
    return PitchConfidenceResult(
        notes=note_events,
        pitched_ratio=pitched_ratio,
        should_trigger_fallback=pitched_ratio < pitched_ratio_threshold,
        used_fallback=False,
        pitched_notes=note_events,
    )


def detect_rhythm_fallback(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fallback_pitch: int = DEFAULT_FALLBACK_PITCH,
    hop_length: int = DEFAULT_ONSET_HOP_LENGTH,
) -> list[NoteEvent]:
    """Use onset detection to derive rhythm-only note events for noisy clips."""
    waveform = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    if waveform.size == 0 or sample_rate <= 0 or librosa is None:
        return []

    try:
        onset_envelope = librosa.onset.onset_strength(
            y=waveform.astype(np.float64),
            sr=sample_rate,
            hop_length=hop_length,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
        )
    except Exception:
        return []

    audio_duration = waveform.shape[0] / float(sample_rate)
    if onset_frames.size == 0:
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if rms <= 1e-4:
            return []
        onset_frames = np.array([0], dtype=int)

    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)
    onset_times = np.clip(onset_times.astype(np.float64, copy=False), 0.0, audio_duration)
    onset_strengths = onset_envelope[onset_frames] if onset_envelope.size else np.array([], dtype=np.float64)
    max_strength = float(np.max(onset_strengths)) if onset_strengths.size else 0.0
    median_interval = (
        float(np.median(np.diff(onset_times)))
        if onset_times.size > 1
        else min(0.25, max(audio_duration, 0.05))
    )

    events: list[NoteEvent] = []
    for index, onset_time in enumerate(onset_times):
        if index + 1 < onset_times.size:
            duration = max(1.0 / sample_rate, float(onset_times[index + 1] - onset_time))
        else:
            remaining = max(1.0 / sample_rate, float(audio_duration - onset_time))
            duration = max(1.0 / sample_rate, min(remaining, median_interval))

        strength = float(onset_strengths[index]) if index < onset_strengths.size else 0.0
        if max_strength > 0:
            confidence = DEFAULT_RHYTHM_CONFIDENCE_FLOOR + (
                (1.0 - DEFAULT_RHYTHM_CONFIDENCE_FLOOR) * (strength / max_strength)
            )
        else:
            confidence = DEFAULT_RHYTHM_CONFIDENCE_FLOOR
        confidence = max(0.0, min(1.0, confidence))

        events.append(
            NoteEvent(
                pitch=fallback_pitch,
                onset=float(onset_time),
                duration=duration,
                velocity=confidence_to_velocity(confidence),
                confidence=confidence,
            )
        )

    return events


def _compute_rms_frames(
    audio: np.ndarray,
    hop_samples: int,
    *,
    frame_samples: int = DEFAULT_RMS_FRAME_SAMPLES,
) -> np.ndarray:
    """Compute centred per-frame RMS on the same hop as PESTO."""
    n = audio.shape[0]
    if n == 0 or hop_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    n_frames = max(1, n // hop_samples + 1)
    rms = np.zeros(n_frames, dtype=np.float32)
    half = max(1, frame_samples // 2)
    audio_f64 = audio.astype(np.float64, copy=False)
    for i in range(n_frames):
        center = i * hop_samples
        start = max(0, center - half)
        end = min(n, center + half)
        frame = audio_f64[start:end]
        if frame.size:
            rms[i] = float(np.sqrt(np.mean(np.square(frame))))
    return rms


def _onset_split_passes_filter(
    rms: np.ndarray,
    pitch_hz: np.ndarray,
    onset_frame: int,
    min_frames: int,
    *,
    max_pitch_stddev_semitones: float = DEFAULT_ONSET_FILTER_MAX_PITCH_STDDEV_SEMITONES,
    amplitude_valley_ratio: float = DEFAULT_ONSET_FILTER_AMPLITUDE_VALLEY_RATIO,
) -> bool:
    """Decide whether an onset-driven segment split should be honoured.

    Distinguishes genuine re-articulated notes from release transients,
    passing tones, and overtone spikes by combining two signals:

    1. **Amplitude valley** — the onset frame's RMS must be at most
       ``amplitude_valley_ratio`` of the local max RMS in the few frames
       before it. A real re-articulation has an amplitude dip + rise; a
       release transient is just decay (no preceding rise to dip from).
    2. **Pitch stability** — the pitch in the post-onset window must have
       standard deviation below ``max_pitch_stddev_semitones``. A target
       note settles flat; a passing tone is a slope.

    Both must pass. Failing either means the onset is treated as noise and
    the segment continues unbroken.
    """
    n = len(rms)
    if n == 0 or onset_frame >= n:
        return False

    look_back = max(2, min_frames // 2)
    pre_start = max(0, onset_frame - look_back)
    post_end = min(len(pitch_hz), onset_frame + min_frames)

    # Need enough post-onset frames to actually constitute a note.
    if post_end - onset_frame < min_frames:
        return False

    # Amplitude valley: did the RMS *dip* before this onset?
    pre_rms = rms[pre_start:onset_frame]
    if pre_rms.size == 0:
        return False
    onset_rms = float(rms[onset_frame])
    pre_max = float(np.max(pre_rms))
    if pre_max <= 1e-9:
        return False
    valley_ok = onset_rms <= amplitude_valley_ratio * pre_max

    # Pitch stability: is the post-onset pitch a settled target?
    post_hz = pitch_hz[onset_frame:post_end]
    valid = (post_hz > 0) & np.isfinite(post_hz)
    if int(valid.sum()) < max(2, min_frames // 2):
        return False
    post_midi = 69.0 + 12.0 * np.log2(post_hz[valid] / 440.0)
    stable_ok = float(np.std(post_midi)) <= max_pitch_stddev_semitones

    return valley_ok and stable_ok


def _compute_onset_frames(
    audio: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = DEFAULT_ONSET_HOP_SAMPLES_AT_16K,
    delta: float = DEFAULT_ONSET_DELTA,
) -> np.ndarray:
    """Return librosa onset frame indices aligned to PESTO's 10 ms grid.

    PESTO's frames live on a 10 ms hop. Using ``hop_length=sample_rate*0.010``
    here means a librosa frame index ``f`` corresponds to the same time as
    PESTO's frame ``f``, so we can use librosa onsets as direct boundary
    signals inside :func:`_segment_pitch_frames`.
    """
    if librosa is None or audio.size == 0 or sample_rate <= 0:
        return np.array([], dtype=int)
    try:
        envelope = librosa.onset.onset_strength(
            y=audio.astype(np.float64), sr=sample_rate, hop_length=hop_length
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=envelope,
            sr=sample_rate,
            hop_length=hop_length,
            units="frames",
            backtrack=False,
            delta=delta,
        )
    except Exception:
        return np.array([], dtype=int)
    return np.asarray(onsets, dtype=int)


def _segment_pitch_frames(
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    hop_s: float,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    minimum_note_length_ms: float,
    voicing_threshold: float = DEFAULT_VOICING_THRESHOLD,
    max_jump_semitones: float = DEFAULT_MAX_JUMP_SEMITONES,
    onset_frames: np.ndarray | None = None,
    rms_frames: np.ndarray | None = None,
    onset_filter_enabled: bool = True,
) -> list[NoteEvent]:
    """Collapse frame-wise (Hz, confidence) into note events.

    When ``onset_frames`` is provided, the segmenter additionally splits the
    running segment at each onset frame (provided the segment already meets
    the minimum-length guard). This recovers repeated-pitch notes in legato
    singing, where the pitch trajectory is continuous but amplitude/spectral
    onsets mark the note boundaries. Without onsets the behaviour reduces to
    pitch-jump segmentation only.

    When ``rms_frames`` is also provided and ``onset_filter_enabled`` is True,
    each onset split is gated by :func:`_onset_split_passes_filter` which
    rejects release transients, passing tones, and overtone spikes. Set
    ``onset_filter_enabled=False`` to honour every onset (the unfiltered
    mode used during development to measure the filter's contribution).
    """
    if pitch_hz.shape != confidence.shape or pitch_hz.size == 0:
        return []

    voiced = (
        (confidence > voicing_threshold)
        & (pitch_hz >= minimum_frequency)
        & (pitch_hz <= maximum_frequency)
        & np.isfinite(pitch_hz)
    )
    midi = np.full(pitch_hz.shape, np.nan, dtype=np.float64)
    midi[voiced] = 69.0 + 12.0 * np.log2(pitch_hz[voiced] / 440.0)

    min_frames = max(1, int(np.ceil((float(minimum_note_length_ms) / 1000.0) / hop_s)))
    onset_set: set[int] = (
        {int(f) for f in onset_frames} if onset_frames is not None and len(onset_frames) > 0 else set()
    )

    events: list[NoteEvent] = []
    cur_midi: list[float] = []
    cur_conf: list[float] = []
    cur_start: int | None = None

    def flush(end_frame: int) -> None:
        nonlocal cur_midi, cur_conf, cur_start
        if cur_start is None or len(cur_midi) < min_frames:
            return
        pitch = int(round(float(np.median(cur_midi))))
        if pitch < 0 or pitch > 127:
            return
        onset = cur_start * hop_s
        duration = max(hop_s, (end_frame - cur_start) * hop_s)
        mean_conf = float(np.clip(np.mean(cur_conf), 0.0, 1.0))
        events.append(
            NoteEvent(
                pitch=pitch,
                onset=onset,
                duration=duration,
                velocity=confidence_to_velocity(mean_conf),
                confidence=mean_conf,
            )
        )

    for i, m in enumerate(midi):
        if not np.isfinite(m):
            flush(i)
            cur_midi = []
            cur_conf = []
            cur_start = None
            continue
        if cur_start is None:
            cur_start = i
            cur_midi = [float(m)]
            cur_conf = [float(confidence[i])]
            continue
        # Force a split at strong amplitude/spectral onsets (recovers repeated
        # pitches in legato runs that the pitch-jump rule alone can't see).
        # The optional filter rejects release transients / passing tones /
        # overtone spikes that would otherwise become spurious notes.
        onset_split = (i in onset_set) and (i - cur_start) >= min_frames
        if onset_split and onset_filter_enabled and rms_frames is not None:
            if not _onset_split_passes_filter(rms_frames, pitch_hz, i, min_frames):
                onset_split = False
        if onset_split or abs(m - float(np.median(cur_midi))) > max_jump_semitones:
            flush(i)
            cur_start = i
            cur_midi = [float(m)]
            cur_conf = [float(confidence[i])]
        else:
            cur_midi.append(float(m))
            cur_conf.append(float(confidence[i]))
    flush(len(midi))

    return events


def extract_melody(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float = DEFAULT_MAXIMUM_FREQUENCY,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    onset_delta: float = DEFAULT_ONSET_DELTA,
    onset_filter_enabled: bool = True,
) -> list[NoteEvent]:
    """Extract monophonic note events with PESTO + median-filter segmenter.

    Onset detection is run alongside PESTO and used as an additional split
    signal inside :func:`_segment_pitch_frames` (see its docstring). The
    librosa hop is locked to PESTO's 10 ms grid so frame indices align.

    ``onset_filter_enabled`` (default True) gates each onset split by pitch
    stability and amplitude valley signals to reject release transients,
    passing tones, and overtone spikes. Set False to honour every onset.
    """
    if audio.size == 0 or sample_rate <= 0:
        return []
    waveform = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    pitch_hz, confidence, hop_s = _pesto_extract_frames(waveform, sample_rate)
    librosa_hop = max(1, int(round(sample_rate * hop_s)))
    onset_frames = _compute_onset_frames(
        waveform, sample_rate, hop_length=librosa_hop, delta=onset_delta
    )
    rms_frames = _compute_rms_frames(waveform, librosa_hop) if onset_filter_enabled else None
    return _segment_pitch_frames(
        pitch_hz,
        confidence,
        hop_s,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        minimum_note_length_ms=minimum_note_length_ms,
        onset_frames=onset_frames,
        rms_frames=rms_frames,
        onset_filter_enabled=onset_filter_enabled,
    )


def extract_melody_with_confidence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float = DEFAULT_MAXIMUM_FREQUENCY,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    pitched_ratio_threshold: float = DEFAULT_PITCHED_RATIO_THRESHOLD,
) -> PitchConfidenceResult:
    """Extract note events and derive fallback confidence metadata for noisy clips."""
    audio_duration = 0.0 if sample_rate <= 0 else audio.shape[0] / float(sample_rate)
    notes = extract_melody(
        audio,
        sample_rate,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        minimum_note_length_ms=minimum_note_length_ms,
    )
    return analyse_pitch_confidence(
        notes,
        audio_duration,
        pitched_ratio_threshold=pitched_ratio_threshold,
    )


def extract_melody_or_fallback(
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_frequency: float = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float = DEFAULT_MAXIMUM_FREQUENCY,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    pitched_ratio_threshold: float = DEFAULT_PITCHED_RATIO_THRESHOLD,
) -> PitchConfidenceResult:
    """Return pitched notes when reliable, otherwise rhythm-only fallback notes."""
    confidence_result = extract_melody_with_confidence(
        audio,
        sample_rate,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        minimum_note_length_ms=minimum_note_length_ms,
        pitched_ratio_threshold=pitched_ratio_threshold,
    )
    if not confidence_result.should_trigger_fallback:
        return confidence_result

    fallback_notes = detect_rhythm_fallback(audio, sample_rate)
    return PitchConfidenceResult(
        notes=fallback_notes,
        pitched_ratio=confidence_result.pitched_ratio,
        should_trigger_fallback=True,
        used_fallback=True,
        pitched_notes=confidence_result.notes,
    )
