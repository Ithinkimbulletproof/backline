# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Train a LoRA adapter on a set of reference songs.

Every batch mixes segments of the references with segments of the general training pool,
so the adapter leans towards the references without forgetting what the base model knows.
The reference share follows from how many passes over the reference segments the run
should make (``reference_epochs``), clipped to ``[min_share, max_share]``, unless a fixed
``reference_share`` is set. Every ``eval_every`` steps the adapter writes parts for
reference prompts; once too many of them copy the references, training stops early.
Afterwards the strongest strength whose parts stay clear of copying becomes the
recommended default for the strength slider.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from backline.adapters import lora
from backline.adapters.cache import AdapterInfo, AdapterStore, adapter_key, file_fingerprint
from backline.config import BacklineConfig
from backline.dataset.reader import Sample, SegmentDataset
from backline.model.generate import GenerationRequest, generate
from backline.model.metrics import WHOLE_NOTE_STEPS
from backline.model.train import collate, load_model
from backline.originality.check import OriginalityChecker
from backline.originality.index import OriginalityIndex
from backline.tokenizer.encode import decode
from backline.tokenizer.render import to_score

ProgressCallback = Callable[[int, int, dict[str, float]], None]
SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class AdapterResult:
    info: AdapterInfo
    path: Path
    cached: bool


class AdapterError(ValueError):
    """Raised when an adapter cannot be trained (no usable reference segments...)."""


def reference_share(config: BacklineConfig, n_reference_segments: int) -> float:
    cfg = config.adapter
    if cfg.reference_share is not None:
        return cfg.reference_share
    wanted = cfg.reference_epochs * n_reference_segments / (cfg.steps * cfg.batch_size)
    return float(min(max(wanted, cfg.min_share), cfg.max_share))


class _Mixer:
    """Draws batches with a fixed share of reference segments."""

    def __init__(
        self,
        sets: dict[str, SegmentDataset],
        references: set[str],
        seed: int,
    ) -> None:
        self.sets = sets
        self.rng = random.Random(seed)
        self.reference: list[tuple[str, int]] = []
        self.pool: list[tuple[str, int]] = []
        for split, dataset in sets.items():
            for i, song in enumerate(dataset.index["song"].tolist()):
                if dataset.song_ids[song] in references:
                    self.reference.append((split, i))
                elif split == "train":
                    self.pool.append((split, i))

    def batch(self, size: int, share: float) -> list[Sample]:
        n_ref = max(1, round(size * share)) if self.reference else 0
        if not self.pool:
            n_ref = size
        picks = [self.rng.choice(self.reference) for _ in range(n_ref)]
        picks += [self.rng.choice(self.pool) for _ in range(size - n_ref)]
        return [self.sets[split][i] for split, i in picks]


def train_adapter(
    config: BacklineConfig,
    checkpoint: Path,
    dataset_dir: Path,
    references: Sequence[str],
    store: AdapterStore,
    *,
    index: OriginalityIndex | None = None,
    device: str | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> AdapterResult:
    """Train (or fetch from the cache) the adapter for ``references``."""
    cfg = config.adapter
    refs = sorted(set(references))
    if not refs:
        raise AdapterError("choose at least one reference song")
    base_print = file_fingerprint(checkpoint)
    key = adapter_key(refs, cfg, base_print)
    if store.exists(key) and not force:
        return AdapterResult(store.info(key), store.path(key), cached=True)

    base, state = load_model(checkpoint, device=device)
    sets = {
        split: SegmentDataset(dataset_dir, split, config, augment=True, seed=config.seed)
        for split in SPLITS
        if (dataset_dir / f"{split}.index.npy").is_file()
    }
    vocab = sets["train"].vocab
    if state["vocab"] != vocab.fingerprint():
        raise AdapterError("the base checkpoint and the dataset use different vocabularies")
    mixer = _Mixer(sets, set(refs), config.seed)
    if not mixer.reference:
        raise AdapterError("the references have no usable segments in the dataset")
    share = reference_share(config, len(mixer.reference))

    model = lora.attach(base, cfg)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.0)
    run_device = next(model.parameters()).device
    autocast = run_device.type == "cuda" and config.train.bf16 and torch.cuda.is_bf16_supported()
    checker = OriginalityChecker(index, config) if index is not None else None
    rng = random.Random(config.seed + 1)
    prompts = [
        mixer.reference[rng.randrange(len(mixer.reference))] for _ in range(cfg.reference_prompts)
    ]

    history: list[dict[str, float]] = []
    stopped = False
    step = 0
    for step in range(1, cfg.steps + 1):
        inputs, targets, mask = (
            t.to(run_device) for t in collate(mixer.batch(cfg.batch_size, share), vocab.pad)
        )
        with torch.autocast(run_device.type, dtype=torch.bfloat16, enabled=autocast):
            loss = model.loss(inputs, targets, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        record = {"step": float(step), "loss": float(loss.detach())}
        if checker is not None and prompts and (step % cfg.eval_every == 0 or step == cfg.steps):
            record["reference_red"] = _red_share(model, sets, prompts, refs, checker, config, rng)
            if record["reference_red"] >= cfg.stop_memorization > 0:
                stopped = True
        history.append(record)
        if progress:
            progress(step, cfg.steps, record)
        if stopped:
            break

    strength_check: dict[str, float] = {}
    recommended = cfg.strengths[-1] if cfg.strengths else 1.0
    if checker is not None and prompts:
        for value in cfg.strengths:
            with lora.strength(model, value):
                red = _red_share(model, sets, prompts, refs, checker, config, rng)
            strength_check[f"{value:g}"] = red
            if red < cfg.stop_memorization or cfg.stop_memorization == 0:
                recommended = value
                break
    else:
        recommended = cfg.strengths[1] if len(cfg.strengths) > 1 else 1.0

    folder = store.path(key)
    model.save_pretrained(str(folder))
    info = AdapterInfo(
        key=key,
        references=refs,
        base=base_print,
        steps=step,
        reference_share=round(share, 4),
        stopped_early=stopped,
        recommended_strength=recommended,
        history=history,
        strength_check=strength_check,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    store.save_info(info)
    return AdapterResult(info, folder, cached=False)


@torch.no_grad()
def _red_share(
    model: torch.nn.Module,
    sets: dict[str, SegmentDataset],
    prompts: list[tuple[str, int]],
    references: list[str],
    checker: OriginalityChecker,
    config: BacklineConfig,
    rng: random.Random,
) -> float:
    reds = 0
    profile = config.instruments.bass.active
    was_training = model.training
    model.eval()
    for split, i in prompts:
        dataset = sets[split]
        record = dataset.index[i]
        start = int(record["offset"])
        tokens = np.asarray(dataset.tokens[start : start + int(record["length"])], dtype=np.int64)
        prompt = tokens[: int(record["target"])].tolist()
        meter = decode(prompt, dataset.vocab).meter
        n_bars = int(record["n_bars"])
        request = GenerationRequest(
            prompt,
            n_bars,
            meter[0] * WHOLE_NOTE_STEPS // meter[1],
            (profile.lowest, profile.highest),
        )
        (sequence,) = generate(model, dataset.vocab, request, variants=1, seed=rng.randrange(2**31))
        report = checker.check(
            to_score(decode(sequence, dataset.vocab), config, n_bars), references=references
        )
        reds += report.verdict == "red"
    model.train(was_training)
    return reds / len(prompts)


def load_for_generation(
    config: BacklineConfig,
    checkpoint: Path,
    references: Sequence[str],
    store: AdapterStore,
    *,
    device: str | None = None,
) -> tuple[torch.nn.Module, AdapterInfo | None]:
    """The base model, wrapped with the cached adapter for ``references`` when one exists."""
    base, _ = load_model(checkpoint, device=device)
    if not references:
        return base, None
    key = adapter_key(sorted(set(references)), config.adapter, file_fingerprint(checkpoint))
    if not store.exists(key):
        return base, None
    return lora.load(base, store.path(key)), store.info(key)
