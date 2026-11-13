# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Read a built dataset split, optionally transposing segments on the fly."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from backline.config import BacklineConfig
from backline.dataset.build import INDEX_DTYPE
from backline.tokenizer.augment import Transposer
from backline.tokenizer.vocab import Vocab


@dataclass(frozen=True, slots=True)
class Sample:
    """One training sequence (possibly transposed) and where its loss starts."""

    tokens: npt.NDArray[np.int64]
    target_start: int
    """Loss is computed from this position on (the ``PLAN`` token and everything after)."""
    song: int
    shift: int


class SegmentDataset:
    """Random access to the segments of one split."""

    def __init__(
        self,
        root: Path,
        split: str,
        config: BacklineConfig,
        *,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.vocab = Vocab.from_json((root / "vocab.json").read_text(encoding="utf-8"))
        expected = Vocab.from_config(config).fingerprint()
        if self.vocab.fingerprint() != expected:
            raise ValueError(
                f"dataset {root} was built with a different vocabulary; rebuild it for this config"
            )
        self.index = np.load(root / f"{split}.index.npy")
        if self.index.dtype != INDEX_DTYPE:
            raise ValueError(f"unexpected index format in {root}")
        path = root / f"{split}.tokens"
        size = path.stat().st_size // 2
        self.tokens = (
            np.memmap(path, dtype=np.uint16, mode="r", shape=(size,))
            if size
            else np.zeros(0, np.uint16)
        )
        self.song_ids: list[str] = json.loads((root / "songs.json").read_text(encoding="utf-8"))
        self.augment = augment
        self.transposer = Transposer(self.vocab, config) if augment else None
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Sample:
        record = self.index[i]
        start = int(record["offset"])
        tokens = np.asarray(self.tokens[start : start + int(record["length"])], dtype=np.int64)
        shift = 0
        if self.transposer is not None:
            allowed = self.transposer.allowed(
                (int(record["guitar_lo"]), int(record["guitar_hi"])),
                (int(record["bass_lo"]), int(record["bass_hi"])),
            )
            shift = self.rng.choice(allowed)
            tokens = self.transposer.apply(tokens, shift)
        return Sample(tokens, int(record["target"]), int(record["song"]), shift)
