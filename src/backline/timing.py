# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Rhythmic feel: map straight notation onto the swung grid it is played on.

Tabs often notate a shuffle as straight 8ths plus a "triplet feel" mark. Readers apply the
feel so that every song is stored as played, the same way the synthetic corpus writes a
shuffle with real triplets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from enum import StrEnum
from fractions import Fraction
from typing import Final

from backline.score import TPQ, Score


class Feel(StrEnum):
    """Swing feels as Guitar Pro names them."""

    STRAIGHT = "straight"
    TRIPLET_8 = "triplet8"
    TRIPLET_16 = "triplet16"
    DOTTED_8 = "dotted8"
    DOTTED_16 = "dotted16"
    SCOTTISH_8 = "scottish8"
    SCOTTISH_16 = "scottish16"


# feel -> (length of the swung unit in ticks, share of the pair given to its first note)
_FEELS: Final = {
    Feel.TRIPLET_8: (TPQ // 2, Fraction(2, 3)),
    Feel.TRIPLET_16: (TPQ // 4, Fraction(2, 3)),
    Feel.DOTTED_8: (TPQ // 2, Fraction(3, 4)),
    Feel.DOTTED_16: (TPQ // 4, Fraction(3, 4)),
    Feel.SCOTTISH_8: (TPQ // 2, Fraction(1, 4)),
    Feel.SCOTTISH_16: (TPQ // 4, Fraction(1, 4)),
}


def feel_position(pos: int, feel: Feel, bar_ticks: int | None = None) -> int:
    """Swung position of a straight position ``pos`` (ticks from a beat-aligned origin).

    Pairs of units (two 8ths for an 8th feel) are re-split: the first takes ``share`` of the
    pair. When ``bar_ticks`` is given, an incomplete pair at the end of the bar is left as is.
    """
    if feel is Feel.STRAIGHT:
        return pos
    unit, share = _FEELS[feel]
    pair = 2 * unit
    index, rest = divmod(pos, pair)
    if bar_ticks is not None and (index + 1) * pair > bar_ticks:
        return pos
    first = pair * share
    if rest <= unit:
        shifted = Fraction(rest, unit) * first
    else:
        shifted = first + Fraction(rest - unit, unit) * (pair - first)
    return index * pair + round(shifted)


def swing_tick(tick: int, feel: Feel = Feel.TRIPLET_8) -> int:
    """Swing an absolute tick; valid when bars start on beat boundaries."""
    return feel_position(tick, feel)


def apply_feel(score: Score, feels: Mapping[int, Feel]) -> None:
    """Move every note of ``score`` onto the swung grid of its bar, in place.

    ``feels`` maps 0-based bar indices to their feel; missing bars are straight.
    """
    if not any(f is not Feel.STRAIGHT for f in feels.values()):
        return
    starts = score.bar_starts()
    end = starts[-1]

    def shift(tick: int) -> int:
        if tick >= end:
            return tick
        bar = score.bar_of(tick)
        feel = feels.get(bar, Feel.STRAIGHT)
        bar_ticks = starts[bar + 1] - starts[bar]
        return starts[bar] + feel_position(tick - starts[bar], feel, bar_ticks)

    for track in score.tracks:
        moved = []
        for note in track.notes:
            start, stop = shift(note.start), shift(note.end)
            moved.append(replace(note, start=start, duration=max(stop - start, 1)))
        track.notes = sorted(moved)
