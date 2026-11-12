# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""The token vocabulary, built deterministically from the config.

Time is counted in grid steps of 1/24 of a quarter note (40 ticks), never in seconds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Final

from backline.config import BacklineConfig
from backline.music import DYNAMICS_ORDER, BassRole, ChordQuality, Mode
from backline.score import TPQ

STEP: Final = TPQ // 24
"""Ticks per grid step."""

PITCH_RANGE: Final = (21, 108)
"""Pitches with their own token (piano range covers guitar and bass)."""

DURATION_STEPS: Final[tuple[int, ...]] = (
    *range(1, 25),
    27, 30, 33, 36, 42, 48, 54, 60, 72, 84, 96, 120, 144, 192,
)  # fmt: skip
"""Allowed note lengths in steps; others snap to the nearest one."""

TS_NUMERATORS: Final = tuple(range(1, 17))
TS_DENOMINATORS: Final = (2, 4, 8, 16)

SPECIALS: Final = (
    "PAD",
    "BOS",
    "EOS",
    "SEP",
    "META",
    "GUITAR",
    "PLAN",
    "BASS",
    "DRUMS",
    "BAR",
    "NC",
)
FLAGS: Final = ("PM", "DEAD")
MODES: Final = (Mode.MAJOR, Mode.MINOR, Mode.MIXOLYDIAN, Mode.DORIAN)


def key_token(tonic: int, mode: Mode) -> str:
    return f"KEY_{tonic % 12}_{mode.value}"


class Vocab:
    """Bidirectional token <-> id map."""

    def __init__(self, tokens: Iterable[str]) -> None:
        self.tokens: list[str] = list(tokens)
        self.ids: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        if len(self.ids) != len(self.tokens):
            raise ValueError("duplicate tokens in vocabulary")

    @classmethod
    def from_config(cls, config: BacklineConfig) -> Vocab:
        tok = config.tokenizer
        tokens: list[str] = list(SPECIALS)
        tokens += [key_token(t, m) for m in MODES for t in range(12)]
        tokens += [f"TS_NUM_{n}" for n in TS_NUMERATORS]
        tokens += [f"TS_DEN_{d}" for d in TS_DENOMINATORS]
        tokens += [f"TEMPO_{i}" for i in range(len(tok.tempo_buckets) + 1)]
        tokens += [f"POS_{p}" for p in range(tok.max_positions)]
        tokens += [f"PITCH_{p}" for p in range(PITCH_RANGE[0], PITCH_RANGE[1] + 1)]
        tokens += [f"DUR_{d}" for d in DURATION_STEPS]
        tokens += [f"VEL_{d.value}" for d in DYNAMICS_ORDER]
        tokens += list(FLAGS)
        tokens += [f"DRUM_{c}" for c in config.drums.classes]
        tokens += [f"DEG_{d}" for d in range(12)]
        tokens += [f"QUAL_{q.value}" for q in ChordQuality]
        tokens += [f"ROLE_{r.value}" for r in BassRole]
        return cls(tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self.ids

    def id(self, token: str) -> int:
        try:
            return self.ids[token]
        except KeyError:
            raise KeyError(f"token not in vocabulary: {token}") from None

    def token(self, index: int) -> str:
        return self.tokens[index]

    def ids_with_prefix(self, prefix: str) -> list[int]:
        return [i for i, t in enumerate(self.tokens) if t.startswith(prefix)]

    @property
    def pad(self) -> int:
        return self.ids["PAD"]

    def fingerprint(self) -> str:
        """Short hash identifying the exact vocabulary (stored with datasets and models)."""
        return hashlib.sha1("\n".join(self.tokens).encode(), usedforsecurity=False).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps({"tokens": self.tokens, "fingerprint": self.fingerprint()}, indent=0)

    @classmethod
    def from_json(cls, text: str) -> Vocab:
        data = json.loads(text)
        vocab = cls(data["tokens"])
        if data.get("fingerprint") not in (None, vocab.fingerprint()):
            raise ValueError("vocabulary fingerprint mismatch")
        return vocab


def snap_duration(steps: int) -> int:
    """Nearest allowed duration in steps (at least one step)."""
    steps = max(steps, 1)
    return min(DURATION_STEPS, key=lambda d: (abs(d - steps), d))


def tempo_bucket(bpm: float, edges: tuple[int, ...]) -> int:
    """Index of the tempo bucket ``bpm`` falls into."""
    for index, edge in enumerate(edges):
        if bpm <= edge:
            return index
    return len(edges)
