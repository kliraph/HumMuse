"""
dataset.py — PyTorch Dataset for monophonic melody token sequences.

Loads filtered .mid files (output of data_filter.py), tokenizes them once
at construction time using a trained REMI tokenizer (from tokenizer_setup.py),
and serves overlapping windows for next-token prediction training.

Each item is an (input_ids, target_ids) pair where input_ids[i+1] == target_ids[i]
— i.e. the standard causal-LM shift-by-one objective the MusicTransformer
expects (see music_transformer.py docstring for the loss contract).

Window construction
-------------------
- Each track is wrapped: [BOS] + tokens + [EOS]
- A sliding window of length (window_size + 1) is taken at stride `stride`
- Within each window: input_ids = window[:-1], target_ids = window[1:]
- The final window per track is right-padded with PAD if it falls short,
  ensuring no melody fragment is dropped purely for being on the boundary.

Mirrors the slide_win64_stride8 setup used for BLSTM_Chord_MH, just with
larger window_size (256) and stride (64) appropriate for transformer context.

Usage
-----
    try:
        from .tokenizer_setup import load_tokenizer
    except ImportError:  # pragma: no cover - supports direct script execution
        from tokenizer_setup import load_tokenizer
    from dataset import MelodyDataset

    tk = load_tokenizer("data/tokenizer.json")
    midi_paths = sorted(Path("data/filtered").glob("*.mid"))
    ds = MelodyDataset(midi_paths, tk, window_size=256, stride=64)

    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    for input_ids, target_ids in loader:
        logits = model(input_ids)
        loss = F.cross_entropy(logits.transpose(1, 2), target_ids,
                               ignore_index=tk["PAD_None"])

Note on multiprocessing
-----------------------
Tokenization is done eagerly at __init__ and cached as Python lists of ints.
On Linux (Kaggle), DataLoader workers fork and inherit the cache cheaply.
On Windows local tests, prefer num_workers=0 (workers re-run __init__).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


class MelodyDataset(Dataset):
    """
    Causal-LM Dataset over monophonic melody tokens.

    Parameters
    ----------
    midi_paths : Sequence[Path | str]
        Paths to filtered monophonic .mid files.
    tokenizer : miditok.REMI
        Trained REMI tokenizer (from tokenizer_setup.train_tokenizer).
    window_size : int, default 256
        Length of input_ids / target_ids returned per sample. Must be
        <= MusicTransformerConfig.max_seq_len (default 512).
    stride : int, default 64
        Step between successive window starts within a single track.
        Smaller stride = more samples per track but more correlation
        between consecutive samples.
    min_tokens : int, default 32
        Tracks whose tokenized length (excluding BOS/EOS) is below this
        are skipped entirely. Below this threshold, even one window would
        be mostly padding — low signal, not worth training on.
    pad_token_id : int | None, default None
        Override PAD token id. If None, read from tokenizer["PAD_None"].
        Must equal MusicTransformerConfig.pad_token_id (default 0).
    bos_token_id : int | None, default None
        Override BOS token id. If None, read from tokenizer["BOS_None"].
    eos_token_id : int | None, default None
        Override EOS token id. If None, read from tokenizer["EOS_None"].
    verbose : bool, default True
        Print per-corpus stats (files tokenized, files skipped, total windows).
    """

    def __init__(
        self,
        midi_paths: Sequence[Path | str],
        tokenizer,
        window_size: int = 256,
        stride: int = 64,
        min_tokens: int = 32,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.window_size = window_size
        self.stride = stride
        self.min_tokens = min_tokens

        # Resolve special token ids — default to whatever the tokenizer says
        self.pad_id = pad_token_id if pad_token_id is not None else tokenizer["PAD_None"]
        self.bos_id = bos_token_id if bos_token_id is not None else tokenizer["BOS_None"]
        self.eos_id = eos_token_id if eos_token_id is not None else tokenizer["EOS_None"]

        # Tokenize all files eagerly. Bad files are skipped with a counter.
        self._tracks: list[list[int]] = []     # one int-list per kept track
        n_failed = 0
        n_too_short = 0
        for p in midi_paths:
            try:
                ids = self._tokenize_one(tokenizer, Path(p))
            except Exception:
                n_failed += 1
                continue
            if len(ids) < min_tokens:
                n_too_short += 1
                continue
            # Wrap with BOS / EOS — gives the model meaningful boundary signals
            wrapped = [self.bos_id] + ids + [self.eos_id]
            self._tracks.append(wrapped)

        # Build a flat (track_idx, window_start) index so __getitem__ is O(1).
        # window length we actually slice is (window_size + 1) for the shift.
        slice_len = window_size + 1
        self._index: list[tuple[int, int]] = []
        for t_idx, ids in enumerate(self._tracks):
            n = len(ids)
            if n <= slice_len:
                # One (padded) window from position 0
                self._index.append((t_idx, 0))
                continue
            # Sliding windows: starts at 0, stride, 2*stride, ...
            # Include the final partial window so no token region is dropped.
            last_start = n - slice_len
            starts = list(range(0, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            for s in starts:
                self._index.append((t_idx, s))

        if verbose:
            print(f"MelodyDataset: {len(self._tracks)} tracks tokenized, "
                  f"{n_failed} failed, {n_too_short} skipped (<{min_tokens} tokens), "
                  f"{len(self._index)} windows total "
                  f"(window={window_size}, stride={stride})")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_one(tokenizer, path: Path) -> list[int]:
        """
        Encode a single .mid file to a flat list of token ids.

        miditok REMI returns either a TokSequence or a list[TokSequence]
        (one per track). Since the filter normalizes every file to a single
        instrument, we take the first sequence's ids.
        """
        out = tokenizer(path)
        if isinstance(out, list):
            if not out:
                raise ValueError(f"tokenizer returned empty list for {path}")
            seq = out[0]
        else:
            seq = out
        return list(seq.ids)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        track_idx, start = self._index[idx]
        ids = self._tracks[track_idx]
        slice_len = self.window_size + 1

        window = ids[start : start + slice_len]
        # Right-pad if we hit the end of the track
        if len(window) < slice_len:
            pad = [self.pad_id] * (slice_len - len(window))
            window = window + pad

        window_t = torch.tensor(window, dtype=torch.long)
        input_ids = window_t[:-1]    # (window_size,)
        target_ids = window_t[1:]    # (window_size,)
        return input_ids, target_ids

    # ------------------------------------------------------------------
    # Diagnostics — useful for thesis chapter 5 reporting
    # ------------------------------------------------------------------

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    @property
    def num_windows(self) -> int:
        return len(self._index)

    def token_length_stats(self) -> dict[str, float]:
        """min/median/mean/max of per-track token length (incl. BOS/EOS)."""
        lens = [len(t) for t in self._tracks]
        if not lens:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        median = (lens_sorted[n // 2] if n % 2 == 1
                  else (lens_sorted[n // 2 - 1] + lens_sorted[n // 2]) / 2)
        return {
            "min": float(lens_sorted[0]),
            "median": float(median),
            "mean": sum(lens) / n,
            "max": float(lens_sorted[-1]),
        }


# ---------------------------------------------------------------------------
# Smoke test against the dry-pass corpus + dry-pass tokenizer.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    here = Path(__file__).parent
    tok_path = here / "data" / "tokenizer_dry.json"
    midi_dir = here / "data" / "filtered_test"

    if not tok_path.exists() or not midi_dir.exists():
        print(f"Dry-pass artifacts missing — run tokenizer_setup.py smoke test first.")
        raise SystemExit(0)

    from tokenizer_setup import load_tokenizer
    tk = load_tokenizer(tok_path)
    midi_files = sorted(midi_dir.glob("*.mid"))
    print(f"Loading {len(midi_files)} files from {midi_dir}")

    ds = MelodyDataset(midi_files, tk, window_size=256, stride=64)
    print(f"Token length stats: {ds.token_length_stats()}")

    # Spot-check shapes
    inp, tgt = ds[0]
    assert inp.shape == (256,), inp.shape
    assert tgt.shape == (256,), tgt.shape
    # Causal shift invariant: target at position i == input at position i+1
    # for the first len(window)-1 entries (last is fresh from window[-1])
    assert (tgt[:-1] == inp[1:]).all(), "shift-by-one invariant broken"
    print(f"Item 0 OK: input.shape={tuple(inp.shape)}, target.shape={tuple(tgt.shape)}")
    print(f"  PAD id={ds.pad_id}, BOS id={ds.bos_id}, EOS id={ds.eos_id}")
    print(f"  input[:8]  = {inp[:8].tolist()}")
    print(f"  target[:8] = {tgt[:8].tolist()}")

    # Spot-check DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch_inp, batch_tgt = next(iter(loader))
    assert batch_inp.shape == (8, 256), batch_inp.shape
    assert batch_tgt.shape == (8, 256), batch_tgt.shape
    print(f"DataLoader batch OK: input.shape={tuple(batch_inp.shape)}, "
          f"target.shape={tuple(batch_tgt.shape)}")
