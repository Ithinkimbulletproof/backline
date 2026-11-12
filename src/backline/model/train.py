# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Training loop for the base model.

AdamW with linear warm-up and cosine decay, bf16 autocast on CUDA, gradient accumulation
and clipping. Every evaluation measures the validation loss, quality metrics of generated
validation segments and the memorization rate: the share of train and validation prompts
whose generated parts the originality check marks red. The best checkpoint minimises
``val loss + memorization_weight * memorization rate``. Checkpoints hold the model, the
optimizer, the scheduler, the data position and the random states, so training resumes
exactly where it stopped.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from backline.config import BacklineConfig
from backline.dataset.reader import Sample, SegmentDataset
from backline.model.generate import GenerationRequest, generate
from backline.model.metrics import WHOLE_NOTE_STEPS, segment_metrics
from backline.model.transformer import Backline, ModelShape
from backline.originality.check import OriginalityChecker
from backline.originality.index import OriginalityIndex
from backline.tokenizer.encode import decode
from backline.tokenizer.render import to_score

ProgressCallback = Callable[[int, int, dict[str, float]], None]


def collate(samples: list[Sample], pad: int) -> tuple[Tensor, Tensor, Tensor]:
    """Inputs, next-token targets and the loss mask (target part only, no padding)."""
    length = max(len(s.tokens) for s in samples) - 1
    inputs = torch.full((len(samples), length), pad, dtype=torch.long)
    targets = torch.full((len(samples), length), pad, dtype=torch.long)
    mask = torch.zeros((len(samples), length), dtype=torch.bool)
    for row, sample in enumerate(samples):
        tokens = torch.from_numpy(sample.tokens)
        n = len(tokens) - 1
        inputs[row, :n] = tokens[:-1]
        targets[row, :n] = tokens[1:]
        mask[row, max(sample.target_start - 1, 0) : n] = True
    return inputs, targets, mask


class Batcher:
    """Shuffled passes over a dataset; its position is part of the checkpoint."""

    def __init__(self, dataset: SegmentDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self.order = self._order()

    def _order(self) -> list[int]:
        order = list(range(len(self.dataset)))
        random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def next(self, size: int) -> list[Sample]:
        batch: list[Sample] = []
        while len(batch) < size:
            if self.cursor >= len(self.order):
                self.epoch += 1
                self.cursor = 0
                self.order = self._order()
            batch.append(self.dataset[self.order[self.cursor]])
            self.cursor += 1
        return batch

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def restore(self, state: dict[str, int]) -> None:
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.order = self._order()


@dataclass(slots=True)
class TrainSummary:
    steps: int
    best_step: int
    best_metric: float
    history: list[dict[str, float]] = field(default_factory=list)
    parameters: int = 0


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Trains a base model on a built dataset."""

    def __init__(
        self,
        config: BacklineConfig,
        dataset_dir: Path,
        checkpoint_dir: Path,
        *,
        device: str | None = None,
        index: OriginalityIndex | None = None,
    ) -> None:
        self.config = config
        self.train_cfg = config.train
        self.checkpoint_dir = checkpoint_dir
        self.device = pick_device(device)
        seed = config.seed
        torch.manual_seed(seed)
        self.train_set = SegmentDataset(
            dataset_dir, "train", config, augment=self.train_cfg.augment, seed=seed
        )
        self.val_set = SegmentDataset(dataset_dir, "val", config)
        if len(self.train_set) == 0:
            raise ValueError(f"dataset {dataset_dir} has no training segments")
        self.vocab = self.train_set.vocab
        self.shape = ModelShape.from_config(config.model, len(self.vocab))
        self.model = Backline(self.shape).to(self.device)
        decay = [p for p in self.model.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.train_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.train_cfg.lr,
            betas=(0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)
        self.batcher = Batcher(self.train_set, seed)
        self.step = 0
        self.best_metric = math.inf
        self.best_step = 0
        self.checker = OriginalityChecker(index, config) if index is not None else None
        self.autocast = (
            self.device.type == "cuda" and self.train_cfg.bf16 and torch.cuda.is_bf16_supported()
        )

    def _lr_factor(self, step: int) -> float:
        warmup = self.train_cfg.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(self.train_cfg.max_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    # ---------------------------------------------------------------- checkpoints

    def save(self, name: str) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "best_metric": self.best_metric,
                "best_step": self.best_step,
                "batcher": self.batcher.state(),
                "shape": asdict(self.shape),
                "vocab": self.vocab.fingerprint(),
                "config": self.config.model_dump(mode="json"),
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "dataset": self.train_set.rng.getstate(),
                },
            },
            path,
        )
        return path

    def resume(self) -> bool:
        """Continue from ``last.pt`` if it exists."""
        path = self.checkpoint_dir / "last.pt"
        if not path.is_file():
            return False
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["vocab"] != self.vocab.fingerprint():
            raise ValueError("checkpoint was trained with a different vocabulary")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        self.best_metric = state["best_metric"]
        self.best_step = state["best_step"]
        self.batcher.restore(state["batcher"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.train_set.rng.setstate(rng["dataset"])
        return True

    # ---------------------------------------------------------------- training

    def _loss(self, samples: list[Sample]) -> Tensor:
        inputs, targets, mask = (t.to(self.device) for t in collate(samples, self.vocab.pad))
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            return self.model.loss(inputs, targets, mask)

    def train(
        self, *, resume: bool = True, progress: ProgressCallback | None = None
    ) -> TrainSummary:
        if resume:
            self.resume()
        cfg = self.train_cfg
        log_dir = self.checkpoint_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = _writer(log_dir)
        history: list[dict[str, float]] = []
        self.model.train()
        started = time.perf_counter()
        tokens_seen = 0
        while self.step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(cfg.grad_accum):
                samples = self.batcher.next(cfg.batch_size)
                tokens_seen += sum(len(s.tokens) for s in samples)
                loss = self._loss(samples) / cfg.grad_accum
                loss.backward()  # type: ignore[no-untyped-call]
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            record: dict[str, float] = {
                "step": float(self.step),
                "train_loss": total,
                "lr": float(self.scheduler.get_last_lr()[0]),
            }
            elapsed = max(time.perf_counter() - started, 1e-9)
            record["tokens_per_second"] = tokens_seen / elapsed
            if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                record |= self.evaluate()
                metric = record["val_loss"] + cfg.memorization_weight * record.get(
                    "memorization", 0.0
                )
                record["selection_metric"] = metric
                if metric < self.best_metric:
                    self.best_metric, self.best_step = metric, self.step
                    self.save("best")
            if self.step % cfg.checkpoint_every == 0 or self.step == cfg.max_steps:
                self.save("last")
            _log(writer, log_dir, record)
            history.append(record)
            if progress:
                progress(self.step, cfg.max_steps, record)
        if writer is not None:
            writer.close()
        return TrainSummary(
            self.step, self.best_step, self.best_metric, history, self.model.parameter_count()
        )

    # ---------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        cfg = self.train_cfg
        self.model.eval()
        losses = []
        rng = random.Random(self.step)
        for _ in range(cfg.eval_batches if len(self.val_set) else 0):
            picks = [self.val_set[rng.randrange(len(self.val_set))] for _ in range(cfg.batch_size)]
            losses.append(float(self._loss(picks)))
        result: dict[str, float] = {"val_loss": float(np.mean(losses)) if losses else float("nan")}
        prompts = cfg.memorization_prompts
        if prompts:
            result |= self._generation_metrics(rng, prompts)
        self.model.train()
        return result

    def _generation_metrics(self, rng: random.Random, prompts: int) -> dict[str, float]:
        reds = longest = checked = 0.0
        adherence, coupling, valid = [], [], []
        for dataset in (self.train_set, self.val_set):
            if len(dataset) == 0:
                continue
            for _ in range(prompts):
                i = rng.randrange(len(dataset))
                record = dataset.index[i]
                tokens = np.asarray(
                    dataset.tokens[
                        int(record["offset"]) : int(record["offset"]) + int(record["length"])
                    ],
                    dtype=np.int64,
                ).tolist()
                prompt = tokens[: int(record["target"])]
                header = decode(prompt, self.vocab)
                n_bars = int(record["n_bars"])
                bar_steps = header.meter[0] * WHOLE_NOTE_STEPS // header.meter[1]
                profile = self.config.instruments.bass.active
                request = GenerationRequest(
                    prompt, n_bars, bar_steps, (profile.lowest, profile.highest)
                )
                (sequence,) = generate(
                    self.model,
                    self.vocab,
                    request,
                    variants=1,
                    seed=rng.randrange(2**31),
                    temperature=self.config.generation.temperature,
                    top_p=self.config.generation.top_p,
                )
                decoded = decode(sequence, self.vocab)
                metrics = segment_metrics(decoded, self.config, n_bars)
                if dataset is self.val_set:
                    adherence.append(metrics.plan_adherence)
                    coupling.append(metrics.bass_kick_coupling)
                    valid.append(float(metrics.valid))
                if self.checker is not None:
                    report = self.checker.check(to_score(decoded, self.config, n_bars))
                    reds += report.verdict == "red"
                    longest += max((p.corpus.longest_bars for p in report.parts), default=0.0)
                    checked += 1
        result: dict[str, float] = {}
        if adherence:
            result |= {
                "plan_adherence": float(np.mean(adherence)),
                "bass_kick_coupling": float(np.mean(coupling)),
                "validity": float(np.mean(valid)),
            }
        if checked:
            result |= {"memorization": reds / checked, "longest_copy_bars": longest / checked}
        return result


def _writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # tensorboard is optional
        return None
    return SummaryWriter(str(log_dir))


def _log(writer: Any, log_dir: Path, record: dict[str, float]) -> None:
    with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if writer is not None:
        step = int(record["step"])
        for key, value in record.items():
            if key != "step" and isinstance(value, float) and math.isfinite(value):
                writer.add_scalar(key, value, step)


def load_model(path: Path, device: str | None = None) -> tuple[Backline, dict[str, Any]]:
    """A trained model and its checkpoint metadata."""
    state = torch.load(path, map_location=pick_device(device), weights_only=False)
    model = Backline(ModelShape(**state["shape"])).to(pick_device(device))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state
