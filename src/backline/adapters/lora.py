# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Attach, save, load and scale LoRA adapters on the Backline transformer (via peft)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from peft.helpers import rescale_adapter_scale
from torch import nn

from backline.config import AdapterConfig
from backline.model.transformer import Backline


def attach(model: Backline, config: AdapterConfig) -> PeftModel:
    """Wrap a base model with fresh, trainable LoRA layers; base weights stay frozen."""
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
    )
    return get_peft_model(model, lora)


def load(model: Backline, folder: Path) -> PeftModel:
    """A base model with a saved adapter on top, ready for generation."""
    wrapped = PeftModel.from_pretrained(model, str(folder))
    wrapped.eval()
    return wrapped


@contextmanager
def strength(model: nn.Module, value: float) -> Iterator[None]:
    """Scale the adapter's effect: 0 is the base model, 1 the adapter as trained."""
    if not isinstance(model, PeftModel) or value == 1.0:
        yield
        return
    with rescale_adapter_scale(model, value):
        yield
