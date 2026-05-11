"""
inference.py — MelodyContinuationModel: load trained checkpoint, generate
candidate continuations from a prompt MIDI.

Sits inside the Phase 5 melody continuation pipeline:

    prompt MIDI ─► MelodyContinuationModel.generate_candidates()
                ─► [{midi_path, tokens, temperature, log_prob}, ...]
                ─► constraint filter (Task 5.4, separate file)
                ─► JS scoring vs. MelodyProfile (Task 5.5)
                ─► top 3 returned via FastAPI

What this module does
---------------------
- Loads a slim "best.pt" checkpoint (saved by train.ipynb's save_inference)
- Loads the matching REMI tokenizer (from tokenizer_setup.load_tokenizer)
- Encodes a prompt MIDI to tokens, prepends BOS, generates N candidates
  spread across a temperature schedule
- Decodes each candidate's *generated* tokens back to a single MIDI file
- Computes per-candidate average log-probability (mean log-prob per
  generated token — length-invariant, comparable across candidates)

What this module does NOT do
----------------------------
- No constraint filtering (Task 5.4)
- No Jensen-Shannon scoring against MelodyProfile (Task 5.5)
- No emotion conditioning — temperature is a *generator parameter*, not a
  bake-in. The FastAPI wrapper (Task 5.7) chooses the temperature schedule
  based on the user's emotion target.
- No chord conditioning at generate time (chord consonance is a downstream
  constraint on candidates, per the design discussion in README.md).
"""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Load .env (PYTHONIOENCODING=utf-8 etc.) at import time so any caller
# of this module gets the same environment guarantees as the training scripts.
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
from miditok import TokSequence

try:
    from .music_transformer import MusicTransformer, MusicTransformerConfig
    from .tokenizer_setup import load_tokenizer
except ImportError:  # pragma: no cover - supports direct script execution
    from music_transformer import MusicTransformer, MusicTransformerConfig
    from tokenizer_setup import load_tokenizer


class MelodyContinuationModel:
    """
    Wraps a trained Music Transformer for FastAPI consumption.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to the slim inference checkpoint written by train.ipynb's
        save_inference(): a torch dict containing {model, config, vocab_size, pad_id}.
    tokenizer_path : Path | str
        Path to the trained REMI tokenizer JSON (tokenizer_setup.load_tokenizer).
    device : str | torch.device, default "cpu"
        Where to run inference. Phase 5 deployment target is CPU; CUDA is
        only used during development for sanity checks.
    top_k : int, default 40
        Top-k filter applied at every sampling step inside
        MusicTransformer.generate(). Matches the model's smoke-test default.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        tokenizer_path: Path | str,
        device: str | torch.device = "cpu",
        top_k: int = 40,
    ):
        self.device = torch.device(device)
        self.top_k = top_k

        # --- tokenizer ----------------------------------------------------
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.pad_id = self.tokenizer["PAD_None"]
        self.bos_id = self.tokenizer["BOS_None"]
        self.eos_id = self.tokenizer["EOS_None"]

        # --- model + checkpoint ------------------------------------------
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Slim checkpoint contract from train.ipynb: dict with these keys
        cfg_dict = ckpt["config"]
        config = MusicTransformerConfig(**cfg_dict)

        # Sanity: tokenizer vocab must match what the model was trained against
        if config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={config.vocab_size} does not match "
                f"tokenizer vocab_size={self.tokenizer.vocab_size}. "
                f"You probably loaded the wrong tokenizer JSON."
            )

        self.model = MusicTransformer(config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        prompt_midi_path: Path | str,
        n_candidates: int = 8,
        max_new_tokens: int = 128,
        temperatures: Sequence[float] = (0.8, 0.9, 1.0),
        out_dir: Path | str | None = None,
        model_id: str = "single",
    ) -> list[dict]:
        """
        Generate N continuation candidates from a prompt MIDI file.

        Candidates are distributed across the given temperatures using a
        ceiling allocation: `n_candidates=8, len(temperatures)=3` yields
        3+3+2 candidates (lower temps slightly favoured — conservative
        continuations are usually a safer default).

        Parameters
        ----------
        prompt_midi_path : Path | str
            The user's primer melody. Will be tokenized as-is; if it carries
            polyphony or drums the tokenizer will encode whatever the file
            contains. Phase 5 expects a monophonic primer (Task 5.4 owns
            input-side validation).
        n_candidates : int, default 8
            Total number of continuations to return.
        max_new_tokens : int, default 128
            Upper bound on tokens generated per candidate. Sampling stops
            earlier if the model emits EOS.
        temperatures : Sequence[float], default (0.8, 0.9, 1.0)
            Sampling temperatures. Distributed across n_candidates.
        out_dir : Path | str | None, default None
            Where to write the candidate .mid files. If None, a fresh
            tempfile directory is created (caller is responsible for cleanup).
        model_id : str, default "single"
            Identifier for the generating model. Included in each returned
            candidate and used as the MIDI filename prefix so multiple models
            can safely share the same output directory.

        Returns
        -------
        list[dict]
            One dict per candidate, in temperature-block order:
            {
                "midi_path":   str,           # absolute path to continuation .mid
                "tokens":      list[int],     # generated tokens only (prompt excluded)
                "temperature": float,
                "log_prob":    float,         # mean log-prob per generated token
                "model_id":     str,           # generating model identifier
            }
        """
        if n_candidates <= 0:
            return []
        if not temperatures:
            raise ValueError("temperatures must be non-empty")

        # --- distribute candidates across temperatures (ceiling) ---------
        per_temp = self._allocate(n_candidates, len(temperatures))

        # --- prepare output directory ------------------------------------
        if out_dir is None:
            out_dir = Path(tempfile.mkdtemp(prefix="melody_cands_"))
        else:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        # --- encode prompt once (constant across all candidates) ---------
        prompt_ids = self._encode_prompt(Path(prompt_midi_path))
        # Truncate prompt if needed to leave room for max_new_tokens
        ctx_budget = self.config.max_seq_len - max_new_tokens - 1  # -1 for safety
        if len(prompt_ids) > ctx_budget:
            prompt_ids = prompt_ids[-ctx_budget:]
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len = prompt_tensor.shape[1]

        # --- generate ----------------------------------------------------
        results: list[dict] = []
        cand_idx = 0
        for temp, count in zip(temperatures, per_temp):
            for _ in range(count):
                full_ids = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=self.top_k,
                    eos_token_id=self.eos_id,
                )
                generated = full_ids[0, prompt_len:].tolist()

                if not generated:
                    # Edge case: model produced zero new tokens (immediate EOS)
                    log_prob = 0.0
                    midi_path = out_dir / f"{model_id}_{cand_idx:02d}_t{temp:.2f}.mid"
                    # Write an empty-tokens MIDI by decoding just the BOS-stripped prompt;
                    # downstream constraint filter will likely reject zero-length.
                    self._decode_to_midi([], midi_path)
                else:
                    log_prob = self._avg_log_prob(prompt_tensor, generated)
                    midi_path = out_dir / f"{model_id}_{cand_idx:02d}_t{temp:.2f}.mid"
                    self._decode_to_midi(generated, midi_path)

                results.append({
                    "midi_path": str(midi_path.resolve()),
                    "tokens": generated,
                    "temperature": float(temp),
                    "log_prob": float(log_prob),
                    "model_id": model_id,
                })
                cand_idx += 1

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _allocate(n_candidates: int, n_temps: int) -> list[int]:
        """
        Ceiling allocation: spread n_candidates across n_temps slots so
        that earlier slots get the extra. Sum equals n_candidates.

        Example: _allocate(8, 3) -> [3, 3, 2]
        """
        base = n_candidates // n_temps
        extra = n_candidates - base * n_temps
        return [base + (1 if i < extra else 0) for i in range(n_temps)]

    def _encode_prompt(self, prompt_path: Path) -> list[int]:
        """Encode prompt MIDI to a flat list of token ids, prepended with BOS."""
        out = self.tokenizer(prompt_path)
        seq = out[0] if isinstance(out, list) else out
        ids = list(seq.ids)
        return [self.bos_id] + ids

    def _decode_to_midi(self, token_ids: list[int], out_path: Path) -> None:
        """
        Decode generated BPE-encoded token ids to a .mid file.

        Two miditok-v3 quirks make this non-obvious:

        1. The model's output ids are *BPE-merge* ids (>= base vocab size).
           A naked TokSequence(ids=...) silently treats them as atomic
           ids and fails with KeyError when decode tries to resolve them.
           Setting `are_ids_encoded=True` tells the tokenizer to first
           expand BPE merges back to atomic tokens.

        2. tokenizer.decode(..., output_path=...) is broken on Windows in
           v3.0.4 (returns a "File not found" runtime error). Use the
           returned Score and call Score.dump_midi(str(path)) instead.

        If the generated id sequence contains no NoteOn (e.g. very short
        sample, or all rests/structural tokens), the decoded Score may
        have zero tracks. We write an empty MIDI in that case so the
        caller still sees a real file path; the constraint filter will
        reject it downstream.
        """
        if not token_ids:
            self._write_empty_midi(out_path)
            return
        seq = TokSequence(ids=token_ids, are_ids_encoded=True)
        score = self.tokenizer.decode([seq])
        if not score.tracks or not score.tracks[0].notes:
            self._write_empty_midi(out_path)
            return
        score.dump_midi(str(out_path))

    @staticmethod
    def _write_empty_midi(out_path: Path) -> None:
        """Write a minimal valid MIDI file containing a single empty track."""
        from symusic import Score, Track
        empty = Score(480)
        empty.tracks.append(Track(name="empty", program=0, is_drum=False))
        empty.dump_midi(str(out_path))

    @torch.no_grad()
    def _avg_log_prob(
        self,
        prompt_tensor: torch.Tensor,
        generated: list[int],
    ) -> float:
        """
        Mean log-probability per *generated* token, computed by a single
        teacher-forcing forward pass over [prompt + generated].

        The cost is one extra forward per candidate at L = prompt+gen
        (e.g. ~50–100ms on CPU at L=384). The model's generate() does not
        return logits; we recover them here for scoring.

        Length-invariant by construction (mean rather than sum), so callers
        can compare candidates of different lengths without bias.
        """
        if not generated:
            return 0.0
        gen_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
        full = torch.cat([prompt_tensor, gen_tensor], dim=1)  # (1, L_p + L_g)

        # Truncate to model context if needed (rare, but safe)
        if full.shape[1] > self.config.max_seq_len:
            full = full[:, -self.config.max_seq_len:]

        logits = self.model(full)                              # (1, L, V)
        log_probs = F.log_softmax(logits, dim=-1)              # (1, L, V)

        # We want log P(generated[i] | prompt + generated[:i])
        # logits at position p predicts token p+1. So for each generated
        # token at position k = prompt_len + i (within `full`), we want
        # log_probs at position k-1, gathered at index `generated[i]`.
        prompt_len = full.shape[1] - len(generated)
        # positions whose prediction we want: prompt_len-1 .. full_len-2
        gather_positions = torch.arange(prompt_len - 1, full.shape[1] - 1, device=self.device)
        token_log_probs = log_probs[0, gather_positions, gen_tensor[0]]
        return float(token_log_probs.mean().item())


@dataclass
class EnsembleMember:
    """
    Configuration for one checkpoint participating in continuation ensemble.

    `count` is this member's candidate target. Its temperature schedule is
    split with the same ceiling allocation rule used by generate_candidates().
    """

    checkpoint_path: Path | str
    tokenizer_path: Path | str
    model_id: str
    temperatures: tuple[float, ...]
    count: int

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(self.checkpoint_path)
        self.tokenizer_path = Path(self.tokenizer_path)
        self.temperatures = tuple(float(t) for t in self.temperatures)
        self.count = int(self.count)
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.count < 0:
            raise ValueError("count must be non-negative")
        if self.count > 0 and not self.temperatures:
            raise ValueError("temperatures must be non-empty when count is positive")

    def temperature_counts(self) -> list[int]:
        """Candidate counts per temperature using the model wrapper's ceiling rule."""
        if self.count == 0:
            return [0 for _ in self.temperatures]
        return MelodyContinuationModel._allocate(self.count, len(self.temperatures))

    def temperature_plan(self) -> list[tuple[float, int]]:
        """Pair each scheduled temperature with its allocated candidate count."""
        return list(zip(self.temperatures, self.temperature_counts()))

    @classmethod
    def from_seeds(
        cls,
        tokenizer_path: Path | str,
        seeds: Sequence[tuple[Path | str, str, Sequence[float], int]],
    ) -> list["EnsembleMember"]:
        """
        Build members for the common same-tokenizer ensemble case.

        Each seed is `(checkpoint_path, model_id, temperatures, count)`.
        """
        return [
            cls(
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                model_id=model_id,
                temperatures=temperatures,
                count=count,
            )
            for checkpoint_path, model_id, temperatures, count in seeds
        ]


class EnsembleMelodyContinuationModel:
    """
    Load and hold one MelodyContinuationModel per ensemble member.

    Members must share the same tokenizer signature. This allows multiple
    checkpoints from the same training run to ensemble, while cross-tokenizer
    mixtures fail before any generation work starts.
    """

    def __init__(
        self,
        members: Sequence[EnsembleMember],
        device: str | torch.device = "cpu",
        top_k: int = 40,
    ):
        if not members:
            raise ValueError("members must be non-empty")

        self.members = list(members)
        self.device = torch.device(device)
        self.top_k = top_k
        self._validate_unique_model_ids()
        self.tokenizer_signature = self._validate_tokenizer_compatibility()
        self.models = {
            member.model_id: MelodyContinuationModel(
                checkpoint_path=member.checkpoint_path,
                tokenizer_path=member.tokenizer_path,
                device=self.device,
                top_k=self.top_k,
            )
            for member in self.members
        }

    def generate_candidates(
        self,
        prompt_midi_path: Path | str,
        max_new_tokens: int = 128,
        out_dir: Path | str | None = None,
    ) -> list[dict]:
        """
        Generate candidates from each ensemble member and concatenate results.

        Total candidate count is `sum(member.count for member in members)`.
        Each member owns its candidate split and temperature schedule.
        """
        if out_dir is None:
            out_dir = Path(tempfile.mkdtemp(prefix="melody_ensemble_cands_"))
        else:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        by_member: list[list[dict]] = []
        for member in self.members:
            model = self.models[member.model_id]
            member_results = model.generate_candidates(
                prompt_midi_path,
                n_candidates=member.count,
                max_new_tokens=max_new_tokens,
                temperatures=member.temperatures,
                out_dir=out_dir,
                model_id=member.model_id,
            )
            for candidate in member_results:
                candidate["model_id"] = member.model_id
            by_member.append(member_results)
        return self._round_robin(by_member)

    @staticmethod
    def _round_robin(by_member: list[list[dict]]) -> list[dict]:
        """Interleave member candidate lists and append longer-member leftovers."""
        interleaved: list[dict] = []
        max_len = max((len(member_results) for member_results in by_member), default=0)
        for idx in range(max_len):
            for member_results in by_member:
                if idx < len(member_results):
                    interleaved.append(member_results[idx])
        return interleaved

    def _validate_unique_model_ids(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for member in self.members:
            if member.model_id in seen:
                duplicates.append(member.model_id)
            seen.add(member.model_id)
        if duplicates:
            raise ValueError(f"duplicate ensemble model_id values: {sorted(set(duplicates))}")

    def _validate_tokenizer_compatibility(self) -> tuple[int, int, int, int]:
        baseline_member = self.members[0]
        baseline = self._tokenizer_signature(baseline_member)
        for member in self.members[1:]:
            signature = self._tokenizer_signature(member)
            if signature != baseline:
                raise ValueError(
                    "Ensemble tokenizer mismatch for "
                    f"model_id={member.model_id!r}: expected "
                    f"(vocab_size, pad_id, bos_id, eos_id)={baseline} from "
                    f"model_id={baseline_member.model_id!r}, got {signature}"
                )
        return baseline

    @staticmethod
    def _tokenizer_signature(member: EnsembleMember) -> tuple[int, int, int, int]:
        tokenizer = load_tokenizer(member.tokenizer_path)
        return (
            int(tokenizer.vocab_size),
            int(tokenizer["PAD_None"]),
            int(tokenizer["BOS_None"]),
            int(tokenizer["EOS_None"]),
        )


# ---------------------------------------------------------------------------
# Smoke test — exercises the pipeline end-to-end against the dry-pass
# tokenizer + a freshly-instantiated (random-weight) model. Useful for
# verifying the encode/generate/decode/log-prob plumbing before any real
# training has happened.
# ---------------------------------------------------------------------------

def _run_contract_smoke() -> None:
    """Verify model_id defaults and filename prefixes without dry-pass data."""
    from types import SimpleNamespace

    class _FakeModel:
        def generate(
            self,
            prompt_tensor: torch.Tensor,
            max_new_tokens: int,
            temperature: float,
            top_k: int,
            eos_token_id: int,
        ) -> torch.Tensor:
            generated = torch.tensor([[7, 8]], dtype=torch.long, device=prompt_tensor.device)
            return torch.cat([prompt_tensor, generated[:, :max_new_tokens]], dim=1)

    class _FakeMemberModel:
        def __init__(self, source_model_id: str):
            self.source_model_id = source_model_id

        def generate_candidates(
            self,
            prompt_midi_path: Path | str,
            n_candidates: int,
            max_new_tokens: int,
            temperatures: Sequence[float],
            out_dir: Path | str,
            model_id: str = "single",
        ) -> list[dict]:
            out_dir = Path(out_dir)
            results: list[dict] = []
            cand_idx = 0
            for temp, count in zip(
                temperatures,
                MelodyContinuationModel._allocate(n_candidates, len(temperatures)),
            ):
                for _ in range(count):
                    midi_path = out_dir / f"{model_id}_{cand_idx:02d}_t{temp:.2f}.mid"
                    midi_path.touch()
                    results.append({
                        "midi_path": str(midi_path.resolve()),
                        "tokens": [cand_idx],
                        "temperature": float(temp),
                        "log_prob": -0.1,
                        "model_id": "single",
                    })
                    cand_idx += 1
            return results

    with tempfile.TemporaryDirectory(prefix="melody_contract_") as tmp:
        runner = object.__new__(MelodyContinuationModel)
        runner.device = torch.device("cpu")
        runner.top_k = 40
        runner.eos_id = 2
        runner.config = SimpleNamespace(max_seq_len=16)
        runner.model = _FakeModel()
        runner._encode_prompt = lambda _path: [1]
        runner._avg_log_prob = lambda _prompt_tensor, _generated: -0.123
        runner._decode_to_midi = lambda _tokens, out_path: Path(out_path).touch()

        cands = runner.generate_candidates(
            "prompt.mid",
            n_candidates=3,
            max_new_tokens=2,
            temperatures=(0.8, 1.0),
            out_dir=tmp,
        )

        print("\nContract smoke generated candidates:")
        for i, c in enumerate(cands):
            filename = Path(c["midi_path"]).name
            print(f"  [{i}] model_id={c['model_id']} -> {filename}")

        assert len(cands) == 3, f"expected 3 candidates, got {len(cands)}"
        assert all(c["model_id"] == "single" for c in cands)
        assert all(Path(c["midi_path"]).name.startswith("single_") for c in cands)

        members = EnsembleMember.from_seeds(
            "tokenizer.json",
            [
                ("a.pt", "cool", (0.5, 0.7, 0.9), 3),
                ("b.pt", "warm", (0.6, 0.8, 1.0), 2),
            ],
        )
        assert members[0].temperature_plan() == [(0.5, 1), (0.7, 1), (0.9, 1)]
        assert members[1].temperature_plan() == [(0.6, 1), (0.8, 1), (1.0, 0)]

        ensemble_members = [
            EnsembleMember("a.pt", "tokenizer.json", "A", (0.5, 0.7, 0.9), 3),
            EnsembleMember("b.pt", "tokenizer.json", "B", (0.5, 0.7, 0.9), 3),
            EnsembleMember("c.pt", "tokenizer.json", "C", (0.7, 0.9), 2),
        ]
        ensemble = object.__new__(EnsembleMelodyContinuationModel)
        ensemble.members = ensemble_members
        ensemble.models = {
            member.model_id: _FakeMemberModel(member.model_id)
            for member in ensemble_members
        }
        ensemble_cands = ensemble.generate_candidates(
            "prompt.mid",
            max_new_tokens=2,
            out_dir=tmp,
        )
        counts = Counter(c["model_id"] for c in ensemble_cands)
        temps_by_model = {
            model_id: [c["temperature"] for c in ensemble_cands if c["model_id"] == model_id]
            for model_id in counts
        }
        assert len(ensemble_cands) == 8, f"expected 8 ensemble candidates, got {len(ensemble_cands)}"
        assert counts == {"A": 3, "B": 3, "C": 2}
        assert [c["model_id"] for c in ensemble_cands] == ["A", "B", "C", "A", "B", "C", "A", "B"]
        assert temps_by_model["A"] == [0.5, 0.7, 0.9]
        assert temps_by_model["B"] == [0.5, 0.7, 0.9]
        assert temps_by_model["C"] == [0.7, 0.9]
        print("Contract smoke OK.")


def _write_smoke_prompt_midi(out_path: Path) -> None:
    """Write a tiny monophonic prompt for tokenizer/inference smoke tests."""
    from symusic import Note, Score, Track

    score = Score(480)
    track = Track(name="prompt", program=0, is_drum=False)
    for idx, pitch in enumerate((60, 62, 64, 67)):
        track.notes.append(Note(idx * 480, 360, pitch, 80))
    score.tracks.append(track)
    score.dump_midi(str(out_path))


def _save_fake_inference_checkpoint(
    checkpoint_path: Path,
    tokenizer,
    seed: int,
) -> None:
    """Save a tiny seeded checkpoint with the same contract as training."""
    torch.manual_seed(seed)
    config = MusicTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        pad_token_id=tokenizer["PAD_None"],
    )
    model = MusicTransformer(config)
    torch.save({
        "epoch": 0,
        "val_loss": 99.9,
        "model": model.state_dict(),
        "config": config.__dict__,
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer["PAD_None"],
    }, checkpoint_path)


def _resolve_smoke_tokenizer_path(tmp_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    lakh_tokenizer = repo_root / "storage" / "models" / "Lakh-MT" / "tokenizer.json"
    if lakh_tokenizer.exists():
        return lakh_tokenizer

    fallback_path = tmp_dir / "tokenizer_smoke.json"
    try:
        from .tokenizer_setup import build_tokenizer
    except ImportError:  # pragma: no cover - supports direct script execution
        from tokenizer_setup import build_tokenizer
    build_tokenizer().save(fallback_path)
    return fallback_path


def _run_ensemble_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="melody_ensemble_smoke_") as tmp:
        tmp_dir = Path(tmp)
        tokenizer_path = _resolve_smoke_tokenizer_path(tmp_dir)
        tokenizer = load_tokenizer(tokenizer_path)

        prompt_path = tmp_dir / "prompt.mid"
        _write_smoke_prompt_midi(prompt_path)

        specs = [
            ("A", 101, (0.5, 0.7, 0.9), 3),
            ("B", 202, (0.5, 0.7, 0.9), 3),
            ("C", 303, (0.7, 0.9), 2),
        ]
        seeds: list[tuple[Path, str, Sequence[float], int]] = []
        for model_id, seed, temperatures, count in specs:
            checkpoint_path = tmp_dir / f"fake_{model_id}_inference.pt"
            _save_fake_inference_checkpoint(checkpoint_path, tokenizer, seed)
            seeds.append((checkpoint_path, model_id, temperatures, count))

        members = EnsembleMember.from_seeds(tokenizer_path, seeds)
        ensemble = EnsembleMelodyContinuationModel(members, device="cpu", top_k=1)
        out_dir = tmp_dir / "candidates"
        cands = ensemble.generate_candidates(
            prompt_path,
            max_new_tokens=4,
            out_dir=out_dir,
        )

        counts = Counter(c["model_id"] for c in cands)
        temps_by_model = {
            model_id: [c["temperature"] for c in cands if c["model_id"] == model_id]
            for model_id, _, _, _ in specs
        }
        midi_paths = [Path(c["midi_path"]) for c in cands]
        filenames = [p.name for p in midi_paths]

        print("\nEnsemble smoke breakdown:")
        for model_id, _, _, _ in specs:
            print(f"  {model_id}: count={counts[model_id]} temps={temps_by_model[model_id]}")
        print(f"  order={[c['model_id'] for c in cands]}")
        print(f"  files={filenames}")

        assert len(cands) == 8, f"expected 8 candidates, got {len(cands)}"
        assert counts == {"A": 3, "B": 3, "C": 2}, counts
        assert temps_by_model["A"] == [0.5, 0.7, 0.9]
        assert temps_by_model["B"] == [0.5, 0.7, 0.9]
        assert temps_by_model["C"] == [0.7, 0.9]
        assert all(p.exists() for p in midi_paths), filenames
        assert len(filenames) == len(set(filenames)), filenames
        print("Ensemble smoke test OK.")


if __name__ == "__main__":
    _run_ensemble_smoke()
