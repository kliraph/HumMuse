"""
Music Transformer for monophonic melody continuation.

Architecture: decoder-only transformer with relative positional self-attention
(Huang et al., "Music Transformer", ICLR 2019). Uses the "skewing" trick for
O(L) memory relative attention rather than O(L^2).

Designed for:
- Monophonic melody token sequences (REMI-style tokenization via MidiTok)
- Small scale: ~5M parameters at default config, fits Kaggle T4 training in <4h
- Symmetric methodology with BLSTM_Chord_MH (PyTorch, trained on filtered MIDI subset)

Usage at training time:
    model = MusicTransformer(vocab_size=tokenizer.vocab_size)
    logits = model(input_ids)  # (B, L, vocab_size)
    loss = F.cross_entropy(logits.transpose(1, 2), target_ids, ignore_index=PAD_ID)

Usage at inference time:
    model.eval()
    output_ids = model.generate(prompt_ids, max_new_tokens=256, temperature=0.9, top_k=40)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MusicTransformerConfig:
    vocab_size: int = 512          # set from MidiTok REMI tokenizer at training time
    d_model: int = 256             # matches BLSTM_Chord_MH hidden size for symmetry
    n_layers: int = 4              # decoder layers
    n_heads: int = 4               # attention heads (d_model must be divisible)
    d_ff: int = 1024               # feedforward dim (typically 4 * d_model)
    max_seq_len: int = 512         # max sequence length for relative attention
    dropout: float = 0.1
    pad_token_id: int = 0


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with relative positional encoding.

    Implements the skewing trick from Huang et al. (2018) for memory-efficient
    relative attention: O(L*d) memory instead of O(L^2*d) for the relative
    embedding tensor.

    The relative embedding E_r has shape (max_seq_len, d_head). We compute
    QE_r^T directly (shape B, H, L, max_seq_len), then skew it to align with
    the standard QK^T attention scores.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Relative position embedding: one per relative distance
        # Distances range from -(max_seq_len-1) to 0 (causal model only attends to past)
        self.E_rel = nn.Parameter(torch.randn(max_seq_len, self.d_head) * 0.02)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    @staticmethod
    def _skew(QE_rel: torch.Tensor) -> torch.Tensor:
        """
        Skew the relative attention scores so that position i attends to
        relative offsets [-(L-1), ..., 0] aligned correctly.

        Input  shape: (B, H, L, L)  — QE_rel where last dim indexes E_rel positions
        Output shape: (B, H, L, L)  — aligned so element [..., i, j] = score at
                                       relative position (i - j) for j <= i.

        Pad-and-reshape trick from the Music Transformer paper.
        """
        B, H, L, _ = QE_rel.shape
        # Pad a zero column on the left along the last dim
        padded = F.pad(QE_rel, (1, 0))                 # (B, H, L, L+1)
        # Reshape to (B, H, L+1, L)
        reshaped = padded.reshape(B, H, L + 1, L)
        # Drop the first row, leaving (B, H, L, L) properly aligned
        skewed = reshaped[:, :, 1:, :]
        return skewed

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, L, d_model)
        causal_mask: (L, L) boolean, True where attention is *masked out* (future positions)
        returns:     (B, L, d_model)
        """
        B, L, _ = x.shape
        assert L <= self.max_seq_len, f"seq len {L} exceeds max {self.max_seq_len}"

        # Project to Q, K, V and split into heads: (B, H, L, d_head)
        Q = self.W_q(x).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).reshape(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Standard content-based attention scores: (B, H, L, L)
        QK = torch.matmul(Q, K.transpose(-2, -1))

        # Relative position scores: take last L positions of E_rel (closest to current)
        E_rel_L = self.E_rel[-L:, :]                   # (L, d_head)
        QE = torch.matmul(Q, E_rel_L.transpose(-2, -1))  # (B, H, L, L)
        QE = self._skew(QE)

        # Combined scores, scaled
        scores = (QK + QE) * self.scale

        # Apply causal mask (True positions become -inf)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                    # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: LN -> attn -> residual -> LN -> FFN -> residual."""

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = RelativeMultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), causal_mask))
        x = x + self.ffn(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """
    Decoder-only Music Transformer for symbolic melody continuation.

    At ~5M parameters with the default config, this is the "small" reference
    implementation from Huang et al. (2018), suitable for monophonic melody
    modelling without overwhelming the constraint-filtering layer downstream.
    """

    def __init__(self, config: MusicTransformerConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = MusicTransformerConfig(**kwargs)
        self.config = config

        self.token_emb = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: share token_emb and head weights for parameter efficiency
        self.head.weight = self.token_emb.weight

        # Pre-compute and register causal mask buffer
        mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) long tensor of token IDs
        returns:   (B, L, vocab_size) logits
        """
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        mask = self.causal_mask[:L, :L]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive sampling with optional top-k filtering.

        prompt_ids:    (B, L_prompt) long tensor — the priming melody tokens
        max_new_tokens: number of tokens to generate beyond the prompt
        temperature:   sampling temperature; lower = more conservative
        top_k:         if set, restrict sampling to top-k logits per step
        eos_token_id:  if set, stop generation when produced (per sequence)

        returns: (B, L_prompt + N) where N <= max_new_tokens
        """
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len if needed
            ctx = ids[:, -self.config.max_seq_len:]
            logits = self.forward(ctx)[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k, dim=-1)
                threshold = top_vals[:, -1:].expand_as(logits)
                logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            ids = torch.cat([ids, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test: forward + generate on a small fixture
    config = MusicTransformerConfig(vocab_size=512)
    model = MusicTransformer(config)
    print(f"Parameters: {model.num_parameters():,}")  # expect ~4-5M

    dummy = torch.randint(1, 512, (2, 64))
    logits = model(dummy)
    assert logits.shape == (2, 64, 512), logits.shape
    print(f"Forward OK: {logits.shape}")

    out = model.generate(dummy[:, :16], max_new_tokens=32, temperature=0.9, top_k=40)
    assert out.shape == (2, 48), out.shape
    print(f"Generate OK: {out.shape}")
