# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""The harmonic plan as an editable table: one row per bar, one cell per beat.

A cell holds a chord symbol ("E5", "Am", "A7", "N.C."); an empty cell continues the chord
before it. Every row starts with a chord when read back, because the model writes one at
every bar line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from backline.music import (
    ChordQuality,
    ChordSymbolError,
    Mode,
    chord_symbol,
    function_label,
    parse_chord_symbol,
)
from backline.tokenizer.encode import PlanChord


class PlanTableError(ValueError):
    """A cell of the table cannot be read; ``bar`` and ``beat`` are 0-based."""

    def __init__(self, bar: int, beat: int, message: str) -> None:
        super().__init__(f"такт {bar + 1}, доля {beat + 1}: {message}")
        self.bar = bar
        self.beat = beat


@dataclass(frozen=True, slots=True)
class Cell:
    """A cell for display: the symbol and its function in the key ("" when empty)."""

    symbol: str
    function: str


def plan_to_table(
    plan: Sequence[PlanChord], n_bars: int, beats: int, bar_steps: int, *, flats: bool = False
) -> list[list[str]]:
    """Chord symbols per beat; a chord is written where it starts (and at each bar line)."""
    beat_steps = max(bar_steps // beats, 1)
    table = [["" for _ in range(beats)] for _ in range(n_bars)]
    for chord in sorted(plan, key=lambda c: (c.bar, c.pos)):
        if not 0 <= chord.bar < n_bars:
            continue
        beat = min(chord.pos // beat_steps, beats - 1)
        root, quality = chord.root, chord.quality
        table[chord.bar][beat] = (
            "N.C." if root is None or quality is None else chord_symbol(root, quality, flats=flats)
        )
    return table


def table_to_plan(
    table: Sequence[Sequence[str]], tonic: int, mode: Mode, bar_steps: int
) -> list[PlanChord]:
    """Read an edited table back into plan chords relative to the key."""
    plan: list[PlanChord] = []
    current: tuple[int, ChordQuality] | str | None = "unset"
    for bar, row in enumerate(table):
        beats = max(len(row), 1)
        beat_steps = max(bar_steps // beats, 1)
        for beat, text in enumerate(row):
            if not text.strip():
                if beat > 0:
                    continue
                if current == "unset":
                    raise PlanTableError(bar, beat, "в первом такте нужен аккорд на первой доле")
                parsed = current
            else:
                try:
                    parsed = parse_chord_symbol(text)
                except ChordSymbolError as exc:
                    raise PlanTableError(bar, beat, str(exc)) from exc
            if beat > 0 and parsed == current:
                continue
            current = parsed
            if parsed is None:
                plan.append(PlanChord(bar, beat * beat_steps, None, None, tonic, mode))
            else:
                assert not isinstance(parsed, str)
                root, quality = parsed
                plan.append(
                    PlanChord(bar, beat * beat_steps, (root - tonic) % 12, quality, tonic, mode)
                )
    return plan


def function_cells(table: Sequence[Sequence[str]], tonic: int) -> list[list[Cell]]:
    """The table with each chord's function in the key (T, S, D, bVII...), for display."""
    rows = []
    for row in table:
        cells = []
        for text in row:
            label = ""
            if text.strip():
                try:
                    parsed = parse_chord_symbol(text)
                except ChordSymbolError:
                    label = "?"
                else:
                    label = "" if parsed is None else function_label((parsed[0] - tonic) % 12)
            cells.append(Cell(text, label))
        rows.append(cells)
    return rows
