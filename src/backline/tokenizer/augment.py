# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Transposition augmentation done on token ids, so datasets store every segment once.

Shifting by ``k`` semitones moves every ``PITCH`` token by ``k`` and every key's tonic by
``k``; plan degrees are relative to the key and stay as they are; drums do not move.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from backline.config import BacklineConfig
from backline.tokenizer.vocab import MODES, PITCH_RANGE, Vocab, key_token


class Transposer:
    """Lookup tables mapping token ids to their transposed ids."""

    def __init__(self, vocab: Vocab, config: BacklineConfig) -> None:
        low, high = config.tokenizer.transpose
        self.shifts = range(low, high + 1)
        self.guitar_range = config.instruments.guitar.input_range
        self.bass_range = config.instruments.bass.train_range
        size = len(vocab)
        self.tables: dict[int, npt.NDArray[np.int64]] = {}
        for k in self.shifts:
            table = np.arange(size, dtype=np.int64)
            for pitch in range(PITCH_RANGE[0], PITCH_RANGE[1] + 1):
                moved = pitch + k
                if PITCH_RANGE[0] <= moved <= PITCH_RANGE[1]:
                    table[vocab.id(f"PITCH_{pitch}")] = vocab.id(f"PITCH_{moved}")
            for mode in MODES:
                for tonic in range(12):
                    table[vocab.id(key_token(tonic, mode))] = vocab.id(key_token(tonic + k, mode))
            self.tables[k] = table

    def allowed(self, guitar: tuple[int, int], bass: tuple[int, int]) -> list[int]:
        """Shifts that keep the riff and the bass inside their instrument ranges."""
        result = []
        for k in self.shifts:
            g_ok = self.guitar_range[0] <= guitar[0] + k and guitar[1] + k <= self.guitar_range[1]
            b_ok = self.bass_range[0] <= bass[0] + k and bass[1] + k <= self.bass_range[1]
            p_ok = (
                PITCH_RANGE[0] <= min(guitar[0], bass[0]) + k
                and max(guitar[1], bass[1]) + k <= PITCH_RANGE[1]
            )
            if g_ok and b_ok and p_ok:
                result.append(k)
        return result or [0]

    def apply(self, tokens: npt.NDArray[np.int64], k: int) -> npt.NDArray[np.int64]:
        """Transposed copy of a token array."""
        if k == 0:
            return tokens.copy()
        return self.tables[k][tokens]
