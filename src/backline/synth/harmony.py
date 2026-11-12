# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Chord progression templates written as scale degrees, independent of key."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from backline.music import ChordQuality, Mode


@dataclass(frozen=True, slots=True)
class ChordStep:
    """One chord of a template: offset from the tonic, quality, length in bars."""

    degree: int
    quality: ChordQuality
    bars: Fraction = Fraction(1)


@dataclass(frozen=True, slots=True)
class Progression:
    """A repeating chord cycle in a mode."""

    name: str
    mode: Mode
    steps: tuple[ChordStep, ...]
    blues: bool = False

    @property
    def cycle_bars(self) -> int:
        """Length of one cycle in whole bars."""
        total = sum((step.bars for step in self.steps), Fraction(0))
        if total.denominator != 1:
            raise ValueError(f"progression {self.name} does not fill whole bars")
        return total.numerator

    @property
    def whole_bar_steps(self) -> bool:
        """True when every chord lasts whole bars, so the template fits any meter."""
        return all(step.bars.denominator == 1 for step in self.steps)


def _p(name: str, mode: Mode, *steps: tuple[int, ChordQuality, Fraction | int]) -> Progression:
    return Progression(name, mode, tuple(ChordStep(d, q, Fraction(b)) for d, q, b in steps))


_MAJ, _MIN, _DOM = ChordQuality.MAJOR, ChordQuality.MINOR, ChordQuality.DOMINANT7
_HALF = Fraction(1, 2)

PROGRESSIONS: Final = (
    _p("I-IV-V-I", Mode.MAJOR, (0, _MAJ, 1), (5, _MAJ, 1), (7, _MAJ, 1), (0, _MAJ, 1)),
    _p("I-V-vi-IV", Mode.MAJOR, (0, _MAJ, 1), (7, _MAJ, 1), (9, _MIN, 1), (5, _MAJ, 1)),
    _p("I-bIII-IV-I", Mode.MAJOR, (0, _MAJ, 1), (3, _MAJ, 1), (5, _MAJ, 1), (0, _MAJ, 1)),
    _p("I-IV", Mode.MAJOR, (0, _MAJ, 1), (5, _MAJ, 1)),
    _p("I-bVII-IV-I", Mode.MIXOLYDIAN, (0, _MAJ, 1), (10, _MAJ, 1), (5, _MAJ, 1), (0, _MAJ, 1)),
    _p("bVII-IV-I", Mode.MIXOLYDIAN, (10, _MAJ, _HALF), (5, _MAJ, _HALF), (0, _MAJ, 1)),
    _p("i-bVI-bVII-i", Mode.MINOR, (0, _MIN, 1), (8, _MAJ, 1), (10, _MAJ, 1), (0, _MIN, 1)),
    _p("i-bVII-bVI-bVII", Mode.MINOR, (0, _MIN, 1), (10, _MAJ, 1), (8, _MAJ, 1), (10, _MAJ, 1)),
    _p("i-iv-bVII-i", Mode.MINOR, (0, _MIN, 1), (5, _MIN, 1), (10, _MAJ, 1), (0, _MIN, 1)),
    _p("i-bVII", Mode.MINOR, (0, _MIN, 1), (10, _MAJ, 1)),
    _p("i-IV dorian", Mode.DORIAN, (0, _MIN, 1), (5, _MAJ, 1)),
    Progression(
        "12-bar blues",
        Mode.MAJOR,
        tuple(
            ChordStep(degree, _DOM, Fraction(1)) for degree in (0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7)
        ),
        blues=True,
    ),
)

MODE_FAMILIES: Final = {
    "major": (Mode.MAJOR, Mode.MIXOLYDIAN),
    "minor": (Mode.MINOR, Mode.DORIAN),
}


def progressions_for(family: str, *, whole_bars_only: bool) -> list[Progression]:
    """Templates in a mode family, optionally only those that fit any meter."""
    modes = MODE_FAMILIES[family]
    return [
        p for p in PROGRESSIONS if p.mode in modes and (p.whole_bar_steps or not whole_bars_only)
    ]
