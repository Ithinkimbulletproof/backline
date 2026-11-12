# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Sample plans and parts from the model: several variants at once, nucleus sampling,
grammar-constrained so every variant decodes into valid notes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from backline.model.grammar import Grammar, GrammarState
from backline.model.transformer import Backline
from backline.tokenizer.vocab import Vocab


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """What to continue: the riff prompt (``BOS .. SEP``) and optional forced tokens."""

    prompt: list[int]
    n_bars: int
    bar_steps: int
    bass_range: tuple[int, int]
    forced: list[int] = field(default_factory=list)
    """Tokens that follow the prompt verbatim, e.g. an edited ``PLAN ... BASS``."""


def _top_p(logits: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative - sorted_logits.softmax(dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)


@torch.no_grad()
def generate(
    model: Backline,
    vocab: Vocab,
    request: GenerationRequest,
    *,
    variants: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    constrained: bool = True,
    seed: int | None = None,
) -> list[list[int]]:
    """Full token sequences (prompt, forced tokens, generated tokens up to ``EOS``)."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    grammar = Grammar(
        vocab, n_bars=request.n_bars, bar_steps=request.bar_steps, bass_range=request.bass_range
    )
    start = grammar.accepts(grammar.start(), request.forced)
    states: list[GrammarState] = [start] * variants
    eos = vocab.id("EOS")

    prefix = request.prompt + request.forced
    budget = model.shape.context_length - len(prefix)
    if budget <= 0:
        raise ValueError("the prompt does not leave room to generate")
    was_training = model.training
    model.eval()
    tokens = torch.tensor([prefix] * variants, dtype=torch.long, device=device)
    logits, cache = model(tokens)
    outputs: list[list[int]] = [list(prefix) for _ in range(variants)]
    done = [False] * variants
    masks = torch.zeros(variants, len(vocab), dtype=torch.bool, device=device)
    for _ in range(budget):
        step_logits = logits[:, -1, :].float() / temperature
        if constrained:
            masks.fill_(True)
            for i, state in enumerate(states):
                masks[i, grammar.allowed(state) if not done[i] else [eos]] = False
            step_logits = step_logits.masked_fill(masks, float("-inf"))
        step_logits = _top_p(step_logits, top_p)
        probabilities = step_logits.softmax(dim=-1)
        chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        for i, token in enumerate(chosen.tolist()):
            if done[i]:
                continue
            outputs[i].append(token)
            if constrained:
                states[i] = grammar.advance(states[i], token)
            if token == eos:
                done[i] = True
        if all(done):
            break
        logits, cache = model(chosen.unsqueeze(1), cache)
    model.train(was_training)
    return outputs


def generated_part(sequence: Sequence[int], vocab: Vocab) -> list[int]:
    """Tokens after ``SEP`` (the plan and the parts)."""
    sep = vocab.id("SEP")
    return list(sequence[list(sequence).index(sep) + 1 :]) if sep in sequence else list(sequence)
