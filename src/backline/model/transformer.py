# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""A compact decoder-only transformer: pre-norm RMSNorm, rotary positions, SwiGLU, tied
embeddings, PyTorch's fused attention and a key/value cache for generation.

Linear layers are named ``qkv``, ``proj``, ``gate``, ``up`` and ``down`` so LoRA adapters
can target them by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from backline.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelShape:
    """Sizes of the transformer (also stored in checkpoints)."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    context_length: int
    dropout: float = 0.0

    @classmethod
    def from_config(cls, config: ModelConfig, vocab_size: int) -> ModelShape:
        return cls(
            vocab_size,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.context_length,
            config.dropout,
        )

    @property
    def hidden(self) -> int:
        """SwiGLU width: about 8/3 of the model width, rounded up to a multiple of 64."""
        return int(math.ceil(self.d_model * 8 / 3 / 64) * 64)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


def rotary_tables(
    head_dim: int, length: int, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """Cosine and sine tables for rotary embeddings, shape ``(length, head_dim)``."""
    inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(length, device=device).float(), inverse)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` of shape ``(batch, heads, time, head_dim)``."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos + rotated * sin).type_as(x)


@dataclass(slots=True)
class LayerCache:
    """Keys and values of the tokens seen so far, for one layer."""

    keys: Tensor
    values: Tensor


class Attention(nn.Module):
    """Causal multi-head self-attention with rotary position embeddings."""

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        if shape.d_model % shape.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = shape.n_heads
        self.head_dim = shape.d_model // shape.n_heads
        self.qkv = nn.Linear(shape.d_model, 3 * shape.d_model, bias=False)
        self.proj = nn.Linear(shape.d_model, shape.d_model, bias=False)
        self.dropout = shape.dropout

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, cache: LayerCache | None = None
    ) -> tuple[Tensor, LayerCache]:
        batch, time, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q, k, v = (
            t.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if cache is not None:
            k = torch.cat([cache.keys, k], dim=2)
            v = torch.cat([cache.values, v], dim=2)
        causal = cache is None or time > 1
        if cache is not None and time > 1:
            past = cache.keys.shape[2]
            mask = torch.ones(time, past + time, dtype=torch.bool, device=x.device).tril(past)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal and cache is None,
                dropout_p=self.dropout if self.training else 0.0,
            )
        out = out.transpose(1, 2).reshape(batch, time, -1)
        return self.proj(out), LayerCache(k, v)


class FeedForward(nn.Module):
    """SwiGLU feed-forward layer."""

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.gate = nn.Linear(shape.d_model, shape.hidden, bias=False)
        self.up = nn.Linear(shape.d_model, shape.hidden, bias=False)
        self.down = nn.Linear(shape.hidden, shape.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        hidden: Tensor = F.silu(self.gate(x)) * self.up(x)
        out: Tensor = self.down(hidden)
        return out


class Block(nn.Module):
    """Pre-norm transformer block: attention, then feed-forward, each with a residual."""

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.norm1 = RMSNorm(shape.d_model)
        self.attn = Attention(shape)
        self.norm2 = RMSNorm(shape.d_model)
        self.mlp = FeedForward(shape)
        self.drop = nn.Dropout(shape.dropout)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, cache: LayerCache | None = None
    ) -> tuple[Tensor, LayerCache]:
        attended, new_cache = self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.drop(attended)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, new_cache


class Backline(nn.Module):
    """The language model over Backline tokens."""

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.shape = shape
        self.embed = nn.Embedding(shape.vocab_size, shape.d_model)
        self.blocks = nn.ModuleList(Block(shape) for _ in range(shape.n_layers))
        self.norm = RMSNorm(shape.d_model)
        self.head = nn.Linear(shape.d_model, shape.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        cos, sin = rotary_tables(shape.d_model // shape.n_heads, shape.context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, parameter in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * shape.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        tokens: Tensor,
        cache: list[LayerCache] | None = None,
    ) -> tuple[Tensor, list[LayerCache]]:
        """Logits for every position and the updated per-layer cache."""
        start = cache[0].keys.shape[2] if cache else 0
        end = start + tokens.shape[1]
        if end > self.shape.context_length:
            raise ValueError(
                f"sequence of {end} tokens exceeds the context of {self.shape.context_length}"
            )
        cos = self.cos[start:end].to(tokens.device)  # type: ignore[index]
        sin = self.sin[start:end].to(tokens.device)  # type: ignore[index]
        x = self.embed(tokens)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, cos, sin, cache[i] if cache else None)
            new_cache.append(layer_cache)
        return self.head(self.norm(x)), new_cache

    def loss(self, tokens: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Cross-entropy over masked positions only (the plan and the parts, not the riff)."""
        logits, _ = self(tokens)
        losses = F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), targets.view(-1), reduction="none"
        )
        weights = mask.view(-1).float()
        return (losses * weights).sum() / weights.sum().clamp(min=1.0)
