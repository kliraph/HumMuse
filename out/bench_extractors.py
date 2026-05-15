"""Compare pitch extractors on the user's hum WAV.

Each extractor produces frame-wise f0 (Hz) + voicing confidence.
We then collapse to notes with the same simple segmenter (median-filtered
midi pitch + voicing gate) so any quality difference reflects the f0 stage,
not the segmenter.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure repo on path
sys.path.insert(0, r"C:\Unios\Studies\Masters\Thesis\HumMuse")

HUM_PATH = r"C:\Users\Klira\Downloads\6e98805296313e2d290792dffa6a4ad2f6492894bd10617e5166f2e5.wav"
TARGET_SR = 16_000

# --- Load + resample
audio, sr = sf.read(HUM_PATH)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
if sr != TARGET_SR:
    import librosa as L
    audio = L.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    sr = TARGET_SR
print(f"Loaded hum: dur={len(audio)/sr:.2f}s sr={sr}")

# --- Segmenter (shared)
def hz_to_midi(hz):
    """Convert array of Hz (NaN = unvoiced) -> midi pitch (NaN preserved)."""
    out = np.full_like(hz, np.nan, dtype=np.float64)
    valid = (hz > 0) & np.isfinite(hz)
    out[valid] = 69.0 + 12.0 * np.log2(hz[valid] / 440.0)
    return out

def segment_notes(midi_frames, hop_s, *, min_voiced_frames=3, max_jump_semitones=1.0):
    """Group consecutive voiced frames into notes by quantizing to int midi and
    splitting on jumps > max_jump_semitones or unvoiced gaps."""
    notes = []
    cur_pitches = []
    cur_start = None

    def flush(end_frame):
        if cur_start is None or not cur_pitches:
            return
        if len(cur_pitches) < min_voiced_frames:
            return
        pitch = int(round(float(np.median(cur_pitches))))
        onset = cur_start * hop_s
        duration = (end_frame - cur_start) * hop_s
        notes.append((pitch, onset, duration, float(np.median(cur_pitches))))

    for i, m in enumerate(midi_frames):
        if not np.isfinite(m):
            flush(i)
            cur_pitches = []
            cur_start = None
            continue
        if cur_start is None:
            cur_start = i
            cur_pitches = [m]
            continue
        if abs(m - np.median(cur_pitches)) > max_jump_semitones:
            flush(i)
            cur_start = i
            cur_pitches = [m]
        else:
            cur_pitches.append(m)
    flush(len(midi_frames))
    return notes


def report(name, midi_frames, hop_s, latency_ms):
    notes = segment_notes(midi_frames, hop_s)
    voiced = int(np.isfinite(midi_frames).sum())
    total = len(midi_frames)
    print(f"\n=== {name} ===  latency={latency_ms:.0f}ms  voiced_frames={voiced}/{total} ({voiced/total*100:.0f}%)")
    print(f"  notes: {len(notes)}")
    for pitch, onset, dur, med_midi in notes:
        # midi -> name
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        oct_ = (pitch // 12) - 1
        nm = names[pitch % 12] + str(oct_)
        cents = (med_midi - pitch) * 100
        print(f"    {nm:>4} (midi={pitch})  onset={onset:5.2f}s dur={dur:4.2f}s  med_cents_off={cents:+5.1f}")
    return notes


# --- 1. pYIN (librosa)
import librosa
t0 = time.perf_counter()
f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz("E2"),
    fmax=librosa.note_to_hz("G5"),
    sr=sr,
    frame_length=2048,
    hop_length=160,  # 10ms at 16k
)
pyin_latency = (time.perf_counter() - t0) * 1000
pyin_midi = hz_to_midi(f0_pyin)
pyin_hop_s = 160 / sr
report("pYIN (librosa)", pyin_midi, pyin_hop_s, pyin_latency)


# --- 2. CREPE (TF)
import crepe
t0 = time.perf_counter()
time_axis, freq, confidence, _ = crepe.predict(
    audio, sr, viterbi=True, step_size=10, model_capacity="full", verbose=0
)
crepe_latency = (time.perf_counter() - t0) * 1000
# CREPE always returns a freq; gate with confidence
freq_voiced = np.where(confidence > 0.5, freq, np.nan)
crepe_midi = hz_to_midi(freq_voiced)
crepe_hop_s = 0.010
report("CREPE-full (TF)", crepe_midi, crepe_hop_s, crepe_latency)


# --- 3. torchcrepe
import torch
import torchcrepe
t0 = time.perf_counter()
audio_t = torch.from_numpy(audio).unsqueeze(0)
hop_length = 160  # 10ms at 16k
pitch, periodicity = torchcrepe.predict(
    audio_t,
    sample_rate=sr,
    hop_length=hop_length,
    fmin=80.0,
    fmax=800.0,
    model="full",
    batch_size=512,
    device="cpu",
    return_periodicity=True,
)
torchcrepe_latency = (time.perf_counter() - t0) * 1000
pitch_np = pitch.squeeze(0).numpy()
period_np = periodicity.squeeze(0).numpy()
tc_voiced = np.where(period_np > 0.5, pitch_np, np.nan)
tc_midi = hz_to_midi(tc_voiced)
report("torchcrepe-full", tc_midi, hop_length / sr, torchcrepe_latency)


# --- 4. PESTO
try:
    import pesto
    t0 = time.perf_counter()
    timesteps, pesto_pitches, pesto_conf, pesto_activations = pesto.predict(
        torch.from_numpy(audio), sr, step_size=10.0
    )
    pesto_latency = (time.perf_counter() - t0) * 1000
    # PESTO returns pitch in Hz and timesteps in ms
    pesto_hz_arr = pesto_pitches.numpy()
    pesto_conf_arr = pesto_conf.numpy()
    pesto_hz_voiced = np.where(pesto_conf_arr > 0.5, pesto_hz_arr, np.nan)
    pesto_midi = hz_to_midi(pesto_hz_voiced)
    report("PESTO", pesto_midi, 0.010, pesto_latency)
except Exception as e:
    print(f"\nPESTO failed: {type(e).__name__}: {e}")


# --- 5. Basic Pitch (the current pipeline) for reference
from ml.melody_sketchpad.preprocess import preprocess_audio, DEFAULT_SAMPLE_RATE
from ml.melody_sketchpad.pitch import extract_melody_or_fallback
t0 = time.perf_counter()
processed = preprocess_audio(audio, sr)
bp_result = extract_melody_or_fallback(processed, sr)
bp_latency = (time.perf_counter() - t0) * 1000
print(f"\n=== Basic Pitch (CURRENT, for reference) ===  latency={bp_latency:.0f}ms  pitched_ratio={bp_result.pitched_ratio:.2f}")
print(f"  notes: {len(bp_result.notes)}")
for n in bp_result.notes:
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    oct_ = (n.pitch // 12) - 1
    nm = names[n.pitch % 12] + str(oct_)
    print(f"    {nm:>4} (midi={n.pitch})  onset={n.onset:5.2f}s dur={n.duration:4.2f}s  conf={n.confidence:.2f}")
