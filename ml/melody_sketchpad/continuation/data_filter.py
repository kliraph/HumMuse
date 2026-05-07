"""
data_filter.py — Extract monophonic melody tracks from the Lakh MIDI Dataset.

Walks an LMD root (clean or full layout — both supported) and for each .mid
file: parses with pretty_midi, scans every non-drum instrument track for
strict monophonicity, applies an 8–32 bar length gate, normalizes the program
to 0 (Acoustic Grand), and writes each kept track to <out_dir>/NNNNNN.mid.

A manifest.csv is written alongside for traceability.

Designed as the first stage of the Phase 5 melody continuation pipeline
(symmetric to the chord-data pipeline in ml/chords/). The output of this
script is the training corpus consumed by tokenizer_setup.py and dataset.py.

CLI
---
    python data_filter.py --lmd-root <path> --out-dir <path> \
        [--max-files N] [--min-bars 8] [--max-bars 32] [--min-notes 16]

Notes
-----
- LMD-clean is artist-organized (Artist/Track.mid, sometimes nested two
  levels). LMD-full uses a hex-prefix layout (0/, 1/, ..., f/). The os.walk
  driver here handles either — it just recurses everything under --lmd-root
  and processes any *.mid or *.midi it finds.
- "Strict monophonic" = no note starts before the previous note ends
  (within a 1ms tolerance to absorb float noise from MIDI quantization).
- Length is estimated from pretty_midi.get_downbeats() when a time sig is
  present; otherwise we fall back to (end_time / sec_per_bar) using the
  estimated tempo and assuming 4/4. Tracks where pretty_midi raises during
  estimation are skipped.
- Program normalization: every kept track is rewritten as a single-instrument
  PrettyMIDI with program=0, is_drum=False. This makes the corpus uniform
  for the tokenizer without affecting REMI tokens (REMI is program-agnostic
  for monophonic input) — purely a hygiene step.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pretty_midi


# --- tunables that aren't worth a CLI flag ----------------------------------

OVERLAP_TOLERANCE_S = 1e-3   # notes within 1ms of each other count as sequential
MIN_NOTES_DEFAULT = 16        # below this, the track is too sparse to be useful
PROGRESS_EVERY = 500          # print a heartbeat every N scanned files


# --- result accounting ------------------------------------------------------

@dataclass
class FilterStats:
    files_scanned: int = 0
    files_parse_failed: int = 0
    tracks_scanned: int = 0
    tracks_drum: int = 0
    tracks_polyphonic: int = 0
    tracks_too_few_notes: int = 0
    tracks_too_short: int = 0
    tracks_too_long: int = 0
    tracks_length_unknown: int = 0
    tracks_kept: int = 0
    parse_errors: Counter = field(default_factory=Counter)

    def as_table(self) -> str:
        lines = [
            "=" * 56,
            "FILTER STATISTICS",
            "=" * 56,
            f"  Files scanned          : {self.files_scanned:>8,}",
            f"  Files parse-failed     : {self.files_parse_failed:>8,}",
            f"  Tracks scanned         : {self.tracks_scanned:>8,}",
            f"    -> drum (skipped)    : {self.tracks_drum:>8,}",
            f"    -> polyphonic        : {self.tracks_polyphonic:>8,}",
            f"    -> too few notes     : {self.tracks_too_few_notes:>8,}",
            f"    -> too short (<min)  : {self.tracks_too_short:>8,}",
            f"    -> too long  (>max)  : {self.tracks_too_long:>8,}",
            f"    -> length unknown    : {self.tracks_length_unknown:>8,}",
            f"  TRACKS KEPT            : {self.tracks_kept:>8,}",
            "=" * 56,
        ]
        if self.parse_errors:
            lines.append("Top parse error types:")
            for err, n in self.parse_errors.most_common(5):
                lines.append(f"    {n:>6,}  {err}")
            lines.append("=" * 56)
        return "\n".join(lines)


# --- per-track logic --------------------------------------------------------

def is_strictly_monophonic(notes: list[pretty_midi.Note]) -> bool:
    """
    A track is strictly monophonic if, after sorting by start time, no note
    begins before the previous note ends (within OVERLAP_TOLERANCE_S).

    Empty or single-note inputs return True; the length/note-count gates
    elsewhere will reject those.
    """
    if len(notes) < 2:
        return True
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    for prev, curr in zip(sorted_notes, sorted_notes[1:]):
        if curr.start < prev.end - OVERLAP_TOLERANCE_S:
            return False
    return True


def estimate_bars(pm: pretty_midi.PrettyMIDI, instrument: pretty_midi.Instrument) -> float | None:
    """
    Estimate the bar count of a single instrument track.

    Strategy:
    1. If pretty_midi can compute downbeats (i.e. a time signature exists),
       count downbeats that fall within [first_note.start, last_note.end].
    2. Otherwise, fall back to (duration_seconds / sec_per_bar) assuming 4/4
       at the estimated tempo.

    Returns None if estimation fails.
    """
    if not instrument.notes:
        return None
    t_start = min(n.start for n in instrument.notes)
    t_end = max(n.end for n in instrument.notes)
    duration = t_end - t_start
    if duration <= 0:
        return None

    # Strategy 1: downbeats from time signature
    try:
        downbeats = pm.get_downbeats()
        if len(downbeats) >= 2:
            in_range = [d for d in downbeats if t_start - 1e-3 <= d <= t_end + 1e-3]
            if len(in_range) >= 2:
                # Number of bar-lines spanning the track ≈ bar count
                return float(len(in_range) - 1) + (t_end - in_range[-1]) / (in_range[-1] - in_range[-2])
    except Exception:
        pass  # fall through to tempo-based estimate

    # Strategy 2: tempo-based fallback (assume 4/4)
    try:
        tempo = pm.estimate_tempo()
        if tempo and tempo > 0:
            sec_per_bar = (60.0 / tempo) * 4.0
            return duration / sec_per_bar
    except Exception:
        return None

    return None


def extract_to_normalized_midi(
    instrument: pretty_midi.Instrument,
    source_pm: pretty_midi.PrettyMIDI,
) -> pretty_midi.PrettyMIDI:
    """
    Build a fresh single-instrument PrettyMIDI containing this track,
    with program normalized to 0 (Acoustic Grand) and is_drum=False.

    Tempo and time-signature changes from the source are preserved so
    REMI tokenization still sees correct beat positions.
    """
    new_pm = pretty_midi.PrettyMIDI(
        resolution=source_pm.resolution,
        initial_tempo=source_pm.estimate_tempo() if source_pm.get_tempo_changes()[1].size > 0 else 120.0,
    )
    # Copy time signature changes
    for ts in source_pm.time_signature_changes:
        new_pm.time_signature_changes.append(
            pretty_midi.TimeSignature(ts.numerator, ts.denominator, ts.time)
        )
    # Copy key signature changes
    for ks in source_pm.key_signature_changes:
        new_pm.key_signature_changes.append(
            pretty_midi.KeySignature(ks.key_number, ks.time)
        )
    # Build the normalized instrument: program=0, not drum
    new_inst = pretty_midi.Instrument(program=0, is_drum=False, name="melody")
    new_inst.notes = [
        pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
        for n in instrument.notes
    ]
    new_pm.instruments.append(new_inst)
    return new_pm


# --- driver -----------------------------------------------------------------

def iter_midi_paths(root: Path) -> Iterator[Path]:
    """Yield every .mid / .midi file under root (case-insensitive)."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".mid") or lower.endswith(".midi"):
                yield Path(dirpath) / name


def _detect_next_index(out_dir: Path) -> int:
    """
    Scan out_dir for files matching NNNNNN.mid and return one past the
    highest number. Used by --append mode to extend an existing corpus
    without overwriting prior outputs.
    """
    if not out_dir.exists():
        return 0
    highest = -1
    for p in out_dir.glob("*.mid"):
        stem = p.stem
        if len(stem) == 6 and stem.isdigit():
            n = int(stem)
            if n > highest:
                highest = n
    return highest + 1


def filter_corpus(
    lmd_root: Path,
    out_dir: Path,
    min_bars: float,
    max_bars: float,
    min_notes: int,
    max_files: int | None,
    append: bool = False,
) -> FilterStats:
    """
    Walk lmd_root and write monophonic-passing tracks to out_dir.

    If `append=True`, the manifest is opened in append mode (no header),
    file numbering starts after the highest existing NNNNNN.mid, and
    a Ctrl-C / kill mid-run does not lose previously kept files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = FilterStats()
    manifest_path = out_dir / "manifest.csv"

    if append:
        kept_idx = _detect_next_index(out_dir)
        # Append to manifest. If it doesn't exist yet, write a header so
        # downstream readers always see a header row.
        write_header = not manifest_path.exists()
        mode = "a"
        print(f"Append mode: starting at index {kept_idx}")
    else:
        kept_idx = 0
        write_header = True
        mode = "w"

    with manifest_path.open(mode, newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow([
                "filename", "source_path", "source_track_idx",
                "n_notes", "n_bars_estimated", "tempo_bpm",
                "source_program", "source_track_name",
            ])
            mf.flush()

        for src_path in iter_midi_paths(lmd_root):
            if max_files is not None and stats.files_scanned >= max_files:
                break

            stats.files_scanned += 1
            if stats.files_scanned % PROGRESS_EVERY == 0:
                print(
                    f"  ... scanned {stats.files_scanned:,} files, "
                    f"kept {stats.tracks_kept:,} tracks",
                    flush=True,
                )

            try:
                pm = pretty_midi.PrettyMIDI(str(src_path))
            except Exception as e:
                stats.files_parse_failed += 1
                stats.parse_errors[type(e).__name__] += 1
                continue

            try:
                source_tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size > 0 else 120.0
            except Exception:
                source_tempo = 120.0

            for t_idx, inst in enumerate(pm.instruments):
                stats.tracks_scanned += 1

                if inst.is_drum:
                    stats.tracks_drum += 1
                    continue

                if len(inst.notes) < min_notes:
                    stats.tracks_too_few_notes += 1
                    continue

                if not is_strictly_monophonic(inst.notes):
                    stats.tracks_polyphonic += 1
                    continue

                bars = estimate_bars(pm, inst)
                if bars is None:
                    stats.tracks_length_unknown += 1
                    continue
                if bars < min_bars:
                    stats.tracks_too_short += 1
                    continue
                if bars > max_bars:
                    stats.tracks_too_long += 1
                    continue

                # Track passes — extract, normalize, write.
                try:
                    new_pm = extract_to_normalized_midi(inst, pm)
                    out_name = f"{kept_idx:06d}.mid"
                    out_path = out_dir / out_name
                    new_pm.write(str(out_path))
                except Exception as e:
                    # Don't increment kept counter if write fails; log and move on.
                    stats.parse_errors[f"write:{type(e).__name__}"] += 1
                    continue

                writer.writerow([
                    out_name,
                    str(src_path),
                    t_idx,
                    len(inst.notes),
                    f"{bars:.2f}",
                    f"{source_tempo:.2f}",
                    inst.program,
                    inst.name or "",
                ])
                # Flush after every kept track so a hard kill (SIGTERM /
                # TaskStop / power loss) does not lose trailing rows. This
                # bit me on the LMD-full run — manifest was 107 rows behind
                # the .mid files on disk because csv.writer buffers.
                mf.flush()
                stats.tracks_kept += 1
                kept_idx += 1

    return stats


# --- entrypoint -------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract monophonic melodies from Lakh MIDI Dataset.",
    )
    p.add_argument("--lmd-root", type=Path, required=True,
                   help="Path to extracted LMD-clean (or LMD-full) directory.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write filtered .mid files + manifest.csv.")
    p.add_argument("--min-bars", type=float, default=8.0,
                   help="Reject tracks shorter than this (in bars).")
    p.add_argument("--max-bars", type=float, default=32.0,
                   help="Reject tracks longer than this (in bars).")
    p.add_argument("--min-notes", type=int, default=MIN_NOTES_DEFAULT,
                   help="Reject tracks with fewer than this many notes.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Process at most N source files (for dry runs).")
    p.add_argument("--append", action="store_true",
                   help="Extend an existing --out-dir: numbering continues "
                        "past the highest NNNNNN.mid, manifest is appended. "
                        "Without this, --out-dir's manifest.csv is overwritten "
                        "and numbering restarts at 000000.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.lmd_root.exists():
        print(f"ERROR: --lmd-root does not exist: {args.lmd_root}", file=sys.stderr)
        return 2

    print(f"LMD root      : {args.lmd_root}")
    print(f"Output dir    : {args.out_dir}")
    print(f"Length window : [{args.min_bars}, {args.max_bars}] bars")
    print(f"Min notes     : {args.min_notes}")
    if args.max_files:
        print(f"Max files     : {args.max_files} (dry run)")
    print()

    try:
        stats = filter_corpus(
            lmd_root=args.lmd_root,
            out_dir=args.out_dir,
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            min_notes=args.min_notes,
            max_files=args.max_files,
            append=args.append,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — partial output preserved in --out-dir.", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1

    print()
    print(stats.as_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
