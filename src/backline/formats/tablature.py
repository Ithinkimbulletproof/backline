# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Lay a :class:`~backline.score.Track` out as tablature beats, independent of file format.

Notes get a string and a fret, note boundaries become beat boundaries, held notes continue
as tied notes, and every beat length is split into durations Guitar Pro can notate.
Boundaries are snapped to :data:`GRID`. Strings are numbered the Guitar Pro way:
1 is the highest string.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from backline.score import TPQ, Note, TrackRole

GRID: Final = TPQ // 24
"""Smallest boundary step: 40 ticks covers 32nds, 16th triplets and 64th triplets."""

PERCUSSION_STRINGS: Final = 6
DEFAULT_TUNINGS: Final = {
    TrackRole.GUITAR: (40, 45, 50, 55, 59, 64),
    TrackRole.BASS: (28, 33, 38, 43),
    TrackRole.OTHER: (40, 45, 50, 55, 59, 64),
    TrackRole.DRUMS: (0,) * PERCUSSION_STRINGS,
}


GP_DYNAMICS: Final = {
    "PPP": 15,
    "PP": 31,
    "P": 47,
    "MP": 63,
    "MF": 79,
    "F": 95,
    "FF": 111,
    "FFF": 127,
}
"""Guitar Pro dynamic marks and the MIDI velocity each one plays at."""


def nearest_dynamic(velocity: int) -> str:
    """Guitar Pro dynamic mark closest to a MIDI velocity."""
    return min(GP_DYNAMICS, key=lambda mark: abs(GP_DYNAMICS[mark] - velocity))


class TabLayoutError(ValueError):
    """Raised when notes cannot be laid out as notatable tablature."""


@dataclass(frozen=True, slots=True)
class NoteValue:
    """A notatable duration: base value (4 = quarter), optional dot or triplet."""

    ticks: int
    value: int
    dotted: bool
    triplet: bool
    cost: float


def _duration_table() -> tuple[NoteValue, ...]:
    table = []
    for value in (1, 2, 4, 8, 16, 32, 64):
        plain = TPQ * 4 // value
        table.append(NoteValue(plain, value, False, False, 1.0))
        table.append(NoteValue(plain * 3 // 2, value, True, False, 1.1))
        table.append(NoteValue(plain * 2 // 3, value, False, True, 1.3))
    return tuple(sorted((d for d in table if d.ticks % GRID == 0), key=lambda d: -d.ticks))


_DURATIONS: Final = _duration_table()


@lru_cache(maxsize=4096)
def split_ticks(ticks: int) -> tuple[NoteValue, ...]:
    """Fewest notatable durations that add up to ``ticks`` (a multiple of :data:`GRID`)."""
    if ticks <= 0 or ticks % GRID:
        raise TabLayoutError(f"cannot notate a length of {ticks} ticks")
    best: dict[int, tuple[float, tuple[NoteValue, ...]]] = {0: (0.0, ())}
    for total in range(GRID, ticks + 1, GRID):
        options = [
            (best[total - d.ticks][0] + d.cost, (*best[total - d.ticks][1], d))
            for d in _DURATIONS
            if d.ticks <= total and total - d.ticks in best
        ]
        if options:
            best[total] = min(options, key=lambda o: o[0])
    if ticks not in best:
        raise TabLayoutError(f"cannot notate a length of {ticks} ticks")
    return tuple(sorted(best[ticks][1], key=lambda d: -d.ticks))


def duration_label(ticks: int) -> str:
    """Readable note value: ``1/4``, ``1/8.`` (dotted), ``1/8t`` (triplet) or ``N ticks``."""
    for d in _DURATIONS:
        if d.ticks == ticks:
            return f"1/{d.value}" + ("." if d.dotted else "") + ("t" if d.triplet else "")
    return f"{ticks} ticks"


def snap(tick: int) -> int:
    """Round a tick to the layout grid."""
    return round(tick / GRID) * GRID


@dataclass(slots=True)
class Placed:
    """A note placed on a string, with snapped start and end."""

    start: int
    end: int
    note: Note
    string: int
    fret: int


def place_fretted(notes: Iterable[Note], tuning: tuple[int, ...], *, frets: int) -> list[Placed]:
    """Give every note a string and fret; notes that cannot be fingered are dropped.

    ``tuning`` lists open strings from the lowest up. Chord notes go from low strings to
    high ones, each on the free string with the lowest fret.
    """
    n_strings = len(tuning)
    placed: list[Placed] = []
    sounding: list[Placed] = []
    ordered_notes = sorted(notes)
    i = 0
    while i < len(ordered_notes):
        onset = snap(ordered_notes[i].start)
        group = []
        while i < len(ordered_notes) and snap(ordered_notes[i].start) == onset:
            group.append(ordered_notes[i])
            i += 1
        sounding = [p for p in sounding if p.end > onset]
        busy = {p.string for p in sounding}
        lowest_free = 0
        for note in sorted(group, key=lambda n: n.pitch):
            options = [
                (note.pitch - tuning[idx], idx)
                for idx in range(n_strings)
                if n_strings - idx not in busy and 0 <= note.pitch - tuning[idx] <= frets
            ]
            if not options:
                continue
            fret, idx = min([o for o in options if o[1] >= lowest_free] or options)
            string = n_strings - idx
            busy.add(string)
            lowest_free = idx + 1
            item = Placed(onset, max(snap(note.end), onset + GRID), note, string, fret)
            placed.append(item)
            sounding.append(item)
    return placed


def place_percussion(notes: Iterable[Note], song_end: int) -> list[Placed]:
    """Drum hits sound until the next hit, as they are notated in tabs.

    The fret holds the MIDI percussion note; simultaneous hits take strings 1, 2, ...
    """
    ordered_notes = sorted(notes)
    onsets = sorted({snap(n.start) for n in ordered_notes})
    next_onset = dict(zip(onsets, [*onsets[1:], song_end], strict=True))
    placed = []
    by_onset: dict[int, int] = {}
    for note in ordered_notes:
        onset = snap(note.start)
        string = by_onset.get(onset, 0) + 1
        if string > PERCUSSION_STRINGS:
            continue
        by_onset[onset] = string
        placed.append(Placed(onset, max(next_onset[onset], onset + GRID), note, string, note.pitch))
    return placed


@dataclass(frozen=True, slots=True)
class TabNote:
    """A note inside a beat; ``tie_in``/``tie_out`` link it to the previous/next beat."""

    placed: Placed
    tie_in: bool
    tie_out: bool


@dataclass(frozen=True, slots=True)
class TabBeat:
    """A beat of one notatable duration; no notes means a rest."""

    start: int
    value: NoteValue
    notes: tuple[TabNote, ...]


def layout_bar(placed: list[Placed], bar_start: int, bar_end: int) -> list[TabBeat]:
    """Beats that fill one bar exactly, in a single voice."""
    inside = [p for p in placed if p.start < bar_end and p.end > bar_start]
    bounds = {bar_start, bar_end}
    for p in inside:
        bounds.update(t for t in (p.start, p.end) if bar_start < t < bar_end)
    beats: list[TabBeat] = []
    for seg_start, seg_end in itertools.pairwise(sorted(bounds)):
        active = [p for p in inside if p.start <= seg_start < p.end]
        offset = seg_start
        for k, piece in enumerate(split_ticks(seg_end - seg_start)):
            piece_end = offset + piece.ticks
            notes = tuple(
                TabNote(p, tie_in=p.start < seg_start or k > 0, tie_out=p.end > piece_end)
                for p in active
            )
            beats.append(TabBeat(offset, piece, notes))
            offset = piece_end
    return beats
