# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Beat grid of a score: where the felt beats fall, meter by meter."""

from __future__ import annotations

from dataclasses import dataclass

from backline.score import Score, TimeSignature


def beats_per_bar(ts: TimeSignature) -> int:
    """Felt beats in a bar: compound meters (6/8, 9/8, 12/8) count dotted quarters."""
    if ts.denominator == 8 and ts.numerator % 3 == 0 and ts.numerator > 3:
        return ts.numerator // 3
    return ts.numerator


@dataclass(frozen=True, slots=True)
class Beat:
    """One felt beat."""

    start: int
    end: int
    bar: int
    index: int
    """Position of the beat inside its bar, from 0."""


def beat_grid(score: Score) -> list[Beat]:
    """Every beat of the score in order."""
    starts = score.bar_starts()
    beats = []
    for bar in range(score.n_bars):
        count = beats_per_bar(score.time_signature_at(bar))
        length = starts[bar + 1] - starts[bar]
        for index in range(count):
            begin = starts[bar] + length * index // count
            end = starts[bar] + length * (index + 1) // count
            beats.append(Beat(begin, end, bar, index))
    return beats
