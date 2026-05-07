"""
tokenizer_setup.py — REMI tokenizer wrapper for monophonic melody training.

Thin wrapper around MidiTok v3's REMI tokenizer, configured for a
monophonic single-instrument corpus (the output of data_filter.py).

Why this file exists, even though MidiTok is one import away:
- Locks in tokenizer config decisions in one place so train.ipynb,
  dataset.py, and inference.py all use identical settings.
- Makes the choices explicit and reviewable for the thesis writeup
  (Chapter 3, methodology) — every knob is justified inline.
- Adds a `train_tokenizer(midi_paths, vocab_size, save_path)` convenience
  that does BPE training + JSON save in one call.

Public API
----------
    build_tokenizer() -> REMI                                  # untrained, base vocab only
    train_tokenizer(midi_paths, vocab_size=512, save_path)     # BPE-trained, written to disk
    load_tokenizer(path) -> REMI                               # round-trip from save_path

Contract with MusicTransformer
------------------------------
- PAD must remain at vocab index 0 (config.pad_token_id=0).
- Final vocab_size returned to MusicTransformerConfig is `tokenizer.vocab_size`,
  read AFTER training (BPE expands the base vocab).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from miditok import REMI, TokenizerConfig


# ---------------------------------------------------------------------------
# Configuration — every non-default decision documented inline.
# ---------------------------------------------------------------------------

# Pitch range A0–C8 (MIDI 21–108). Wider than necessary for vocal melody
# (typically C2–C6) but covers any instrument-played monophonic line that
# survives the filter. Same as MidiTok default; kept explicit for clarity.
PITCH_RANGE: tuple[int, int] = (21, 109)

# Beat resolution: 8 subdivisions per beat for the first 4 beats (32nd-note
# grid), 4 for beats 4–12 (16th-note grid). Default. Adequate for the
# rhythmic complexity of pop/folk monophonic melody — we are not modelling
# tuplet-heavy classical or jazz lines here.
BEAT_RES: dict[tuple[int, int], int] = {(0, 4): 8, (4, 12): 4}

# Velocity bins. Default is 32; we drop to 8 because:
#  - monophonic melody has narrower expressive range than polyphonic piano
#  - smaller velocity vocab = fewer rare tokens fighting for BPE merges
#  - 8 bins still cover ppp / pp / p / mp / mf / f / ff / fff
NUM_VELOCITIES: int = 8

# Special tokens — PAD MUST be at index 0 to match
# MusicTransformerConfig.pad_token_id=0. BOS/EOS used by inference.py for
# prompt boundaries and stop conditions. MASK retained from default in case
# we want a masked-LM auxiliary objective later (not used in Phase 5).
SPECIAL_TOKENS: list[str] = ["PAD", "BOS", "EOS", "MASK"]

# REMI feature flags for monophonic melody:
#  - use_chords=False        — corpus is monophonic by construction
#  - use_rests=True          — silence is structural in melody, model must see it
#  - use_tempos=True         — helps model learn rhythmic style families
#  - use_time_signatures=True — helps learn metric structure
#  - use_programs=False      — single-instrument corpus (program normalized to 0)
#  - use_sustain_pedals=False — irrelevant for monophonic
#  - use_pitch_intervals=False — keep token space small; absolute pitches OK at this scale
USE_CHORDS = False
USE_RESTS = True
USE_TEMPOS = True
USE_TIME_SIGS = True
USE_PROGRAMS = False
USE_SUSTAIN = False
USE_PITCH_INTERVALS = False

# Default BPE vocab target. Set to match MusicTransformerConfig.vocab_size=512.
# The base REMI vocab with the above settings comes in around 280–320 tokens;
# remaining ~200 slots are filled with BPE merges learned from the corpus.
DEFAULT_VOCAB_SIZE: int = 512


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tokenizer() -> REMI:
    """
    Construct a fresh, untrained REMI tokenizer with the monophonic-melody
    configuration above. Cheap and idempotent — call as many times as needed.

    Returns
    -------
    REMI
        Configured tokenizer. Base vocab populated; no BPE merges yet.
    """
    config = TokenizerConfig(
        pitch_range=PITCH_RANGE,
        beat_res=BEAT_RES,
        num_velocities=NUM_VELOCITIES,
        special_tokens=SPECIAL_TOKENS,
        use_chords=USE_CHORDS,
        use_rests=USE_RESTS,
        use_tempos=USE_TEMPOS,
        use_time_signatures=USE_TIME_SIGS,
        use_programs=USE_PROGRAMS,
        use_sustain_pedals=USE_SUSTAIN,
        use_pitch_intervals=USE_PITCH_INTERVALS,
    )
    tokenizer = REMI(tokenizer_config=config)
    return tokenizer


def train_tokenizer(
    midi_paths: Sequence[Path | str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    save_path: Path | str | None = None,
) -> REMI:
    """
    Train a BPE tokenizer over the provided MIDI files and (optionally) save it.

    Parameters
    ----------
    midi_paths : Sequence[Path | str]
        Paths to filtered monophonic .mid files (output of data_filter.py).
    vocab_size : int, default 512
        Target BPE vocabulary size. The base REMI vocab is already populated
        from the configuration; BPE merges fill the rest of the budget.
        Must be >= the base vocab size (~280) — train() will raise otherwise.
    save_path : Path | str | None
        If given, write the trained tokenizer to this JSON file. The parent
        directory will be created if missing. Use load_tokenizer() to read back.

    Returns
    -------
    REMI
        The trained tokenizer instance.

    Raises
    ------
    ValueError
        If `midi_paths` is empty or `vocab_size` < base vocab size.
    """
    if not midi_paths:
        raise ValueError("midi_paths is empty — nothing to train BPE on")

    # Normalize paths to Path objects (MidiTok accepts both but is stricter
    # internally on some platforms; pre-converting avoids edge cases).
    paths = [Path(p) for p in midi_paths]

    tokenizer = build_tokenizer()
    base_vocab = tokenizer.vocab_size
    if vocab_size < base_vocab:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than base REMI vocab ({base_vocab}). "
            f"Either raise vocab_size or trim REMI features in tokenizer_setup.py."
        )

    print(f"Training BPE: base vocab={base_vocab}, target={vocab_size}, "
          f"corpus={len(paths)} files")
    tokenizer.train(
        vocab_size=vocab_size,
        model="BPE",
        files_paths=paths,
    )
    print(f"Trained vocab size: {tokenizer.vocab_size}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(save_path)
        print(f"Saved tokenizer to {save_path}")

    return tokenizer


def load_tokenizer(path: Path | str) -> REMI:
    """
    Load a previously trained tokenizer from a JSON file written by
    train_tokenizer(..., save_path=...).

    Parameters
    ----------
    path : Path | str
        Path to the tokenizer JSON file.

    Returns
    -------
    REMI
        The loaded tokenizer, ready for encoding/decoding.

    Raises
    ------
    FileNotFoundError
        If the path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return REMI(params=path)


# ---------------------------------------------------------------------------
# CLI — `python tokenizer_setup.py --midi-dir ... --out ... --vocab-size 512`
# trains BPE on the given corpus and saves the tokenizer JSON. Run with no
# args to fall back to the dry-pass smoke test.
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Build untrained, then train on dry-pass corpus and round-trip."""
    tk = build_tokenizer()
    print(f"Untrained vocab size: {tk.vocab_size}")
    print(f"PAD token id: {tk['PAD_None']}")
    print(f"BOS token id: {tk['BOS_None']}")
    print(f"EOS token id: {tk['EOS_None']}")
    assert tk["PAD_None"] == 0, "PAD must be at index 0 to match MusicTransformerConfig.pad_token_id=0"

    here = Path(__file__).parent
    dry = here / "data" / "filtered_test"
    if not dry.exists():
        print(f"\nDry-pass corpus not found at {dry} — nothing to train.")
        return
    midi_files = sorted(dry.glob("*.mid"))
    if not midi_files:
        print(f"\nNo .mid files in {dry} — nothing to train.")
        return
    print(f"\nFound {len(midi_files)} files in dry-pass corpus — training BPE")
    save_to = here / "data" / "tokenizer_dry.json"
    trained = train_tokenizer(midi_files, vocab_size=512, save_path=save_to)
    reloaded = load_tokenizer(save_to)
    assert reloaded.vocab_size == trained.vocab_size
    print(f"Round-trip OK: reloaded vocab size = {reloaded.vocab_size}")
    sample = midi_files[0]
    seq = reloaded(sample)
    seq = seq if not isinstance(seq, list) else seq[0]
    print(f"\nEncoded {sample.name}: ids[:20] = {seq.ids[:20]}, total = {len(seq.ids)}")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Train a REMI BPE tokenizer on a directory of monophonic .mid files."
    )
    p.add_argument("--midi-dir", type=Path,
                   help="Directory of .mid files to train BPE on. If omitted, runs the dry-pass smoke test.")
    p.add_argument("--out", type=Path, default=None,
                   help="Path to save the trained tokenizer JSON (default: <midi-dir>/../tokenizer.json).")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                   help=f"Target BPE vocab size (default {DEFAULT_VOCAB_SIZE}).")
    args = p.parse_args(argv)

    if args.midi_dir is None:
        _smoke_test()
        return 0

    if not args.midi_dir.exists():
        print(f"ERROR: --midi-dir does not exist: {args.midi_dir}")
        return 2

    midi_files = sorted(args.midi_dir.glob("*.mid"))
    if not midi_files:
        print(f"ERROR: no .mid files found in {args.midi_dir}")
        return 2

    out = args.out if args.out is not None else (args.midi_dir.parent / "tokenizer.json")
    print(f"MIDI dir   : {args.midi_dir}")
    print(f"Files      : {len(midi_files):,}")
    print(f"Vocab size : {args.vocab_size}")
    print(f"Save path  : {out}")
    print()
    train_tokenizer(midi_files, vocab_size=args.vocab_size, save_path=out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
