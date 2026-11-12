# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Pattern generators for the guitar riff, the bass line and the drum groove.

All patterns are written on a straight 8th/16th grid; a shuffle feel is applied to the
whole song afterwards by :func:`backline.timing.swing_tick`. Drum grooves for a shuffle
are written on
the triplet grid directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from backline.music import BassRole, ChordQuality, Dynamic, Mode
from backline.score import TPQ, NoteFlag

Q: Final = TPQ
E: Final = TPQ // 2
S: Final = TPQ // 4
T8: Final = TPQ // 3

GUITAR_LOWEST_ROOT: Final = 40
"""Power-chord roots sit between E2 (40) and D#3 (51)."""


@dataclass(frozen=True, slots=True)
class Span:
    """A chord sounding from ``start`` to ``end`` ticks, with what comes next."""

    start: int
    end: int
    root_pc: int
    degree: int
    quality: ChordQuality
    tonic_pc: int
    mode: Mode
    next_root_pc: int

    @property
    def length(self) -> int:
        return self.end - self.start


# ------------------------------------------------------------------------ guitar


class Shape(StrEnum):
    """Which notes of the chord a guitar hit plays."""

    ROOT = "root"
    FIFTH = "fifth"
    SIXTH = "sixth"
    POWER = "power"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class Hit:
    """A guitar strike: when, how long, which shape, how loud."""

    offset: int
    duration: int
    shape: Shape
    dynamic: Dynamic = Dynamic.NORMAL
    flags: NoteFlag = NoteFlag.NONE


RIFF_STYLES: Final = ("eighths_pm", "chug_stabs", "quarters", "syncopated", "sustain", "boogie")
FULL_CHORD_RIFFS: Final = ("quarters", "syncopated", "sustain", "eighths_pm")


def riff_hits(style: str, span: Span, *, full: bool) -> list[Hit]:
    """Rhythm of the guitar over one chord span."""
    length = span.length
    chord = Shape.FULL if full else Shape.POWER
    eighths = range(0, length, E)
    if style == "eighths_pm":
        if full:
            return [Hit(o, E, chord, Dynamic.ACCENT if o == 0 else Dynamic.NORMAL) for o in eighths]
        return [
            Hit(o, E, chord, Dynamic.ACCENT)
            if o == 0
            else Hit(o, E, chord, Dynamic.NORMAL, NoteFlag.PALM_MUTE)
            for o in eighths
        ]
    if style == "chug_stabs":
        stabs = {0, Q + E}
        return [
            Hit(o, E, chord, Dynamic.ACCENT)
            if o in stabs
            else Hit(o, E, Shape.ROOT, Dynamic.NORMAL, NoteFlag.PALM_MUTE)
            for o in eighths
        ]
    if style == "quarters":
        return [
            Hit(o, Q, chord, Dynamic.ACCENT if o == 0 else Dynamic.NORMAL)
            for o in range(0, length, Q)
        ]
    if style == "syncopated":
        hits: list[Hit] = []
        offset, index = 0, 0
        while offset < length:
            size = min((3 * E, 3 * E, 2 * E)[index % 3], length - offset)
            hits.append(Hit(offset, size, chord, Dynamic.ACCENT if offset == 0 else Dynamic.NORMAL))
            offset += size
            index += 1
        return hits
    if style == "sustain":
        return [Hit(0, length, chord, Dynamic.ACCENT)]
    if style == "boogie":
        return [
            Hit(o, E, Shape.FIFTH if i % 2 == 0 else Shape.SIXTH, Dynamic.NORMAL)
            for i, o in enumerate(eighths)
        ]
    raise ValueError(f"unknown riff style: {style}")


_FULL_VOICINGS: Final = {
    ChordQuality.MAJOR: (0, 7, 12, 16),
    ChordQuality.MINOR: (0, 7, 12, 15),
    ChordQuality.DOMINANT7: (0, 7, 10, 16),
    ChordQuality.MINOR7: (0, 7, 10, 15),
    ChordQuality.MAJOR7: (0, 7, 11, 16),
    ChordQuality.SUS2: (0, 7, 14),
    ChordQuality.SUS4: (0, 7, 17),
    ChordQuality.DIMINISHED: (0, 6, 12, 15),
    ChordQuality.AUGMENTED: (0, 8, 12, 16),
    ChordQuality.POWER: (0, 7, 12),
}

_SHAPE_INTERVALS: Final = {
    Shape.ROOT: (0,),
    Shape.FIFTH: (0, 7),
    Shape.SIXTH: (0, 9),
    Shape.POWER: (0, 7, 12),
}


def guitar_pitches(shape: Shape, root_pc: int, quality: ChordQuality) -> list[int]:
    """Pitches of a guitar hit; the root sits on the low strings."""
    root = GUITAR_LOWEST_ROOT + (root_pc - GUITAR_LOWEST_ROOT) % 12
    intervals = _FULL_VOICINGS[quality] if shape is Shape.FULL else _SHAPE_INTERVALS[shape]
    return [root + i for i in intervals]


# -------------------------------------------------------------------------- bass


@dataclass(frozen=True, slots=True)
class BassHit:
    """A bass note with its harmonic role, known by construction."""

    offset: int
    duration: int
    pitch: int
    role: BassRole
    dynamic: Dynamic = Dynamic.NORMAL


BASS_STYLES_POWER: Final = (
    "root_eighths",
    "root_quarters",
    "root_fifth",
    "follow_riff",
    "approach",
    "neighbor",
)
BASS_STYLES_FULL: Final = (*BASS_STYLES_POWER, "walking")
BASS_STYLES_SOUNDING_THIRD: Final = ("walking", "blues_boogie")

_BOOGIE: Final = ((0, 4, 7, 9, 10, 9, 7, 4), "CCCPCPCC")
_BOOGIE_BLUE: Final = ((0, 3, 4, 7, 9, 10, 9, 7), "CBCCPCPC")
_ROLE_CODES: Final = {
    "C": BassRole.CHORD_TONE,
    "P": BassRole.PASSING,
    "B": BassRole.BLUE_NOTE,
}


def _fit(pitch: int, low: int, high: int) -> int:
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


def bass_hits(
    style: str,
    span: Span,
    *,
    low: int,
    high: int,
    riff: list[Hit],
    blue: bool = False,
) -> list[BassHit]:
    """Bass line over one chord span, within the instrument range ``low..high``."""
    length = span.length
    root = low + (span.root_pc - low) % 12
    fifth = _fit(root + 7, low, high)
    octave = root + 12 if root + 12 <= high else root
    step = (span.next_root_pc - span.root_pc + 6) % 12 - 6
    target = _fit(root + step, low, high)
    approach: int | None = None
    if span.next_root_pc != span.root_pc:
        approach = _fit(target - 1 if target > root else target + 1, low, high)
    scale = {(span.tonic_pc + s) % 12 for s in span.mode.scale}
    neighbor = root + 1 if (root + 1) % 12 in scale else root + 2
    eighths = list(range(0, length, E))
    quarters = list(range(0, length, Q))

    def accent(offset: int) -> Dynamic:
        return Dynamic.ACCENT if offset == 0 else Dynamic.NORMAL

    if style == "root_eighths":
        return [BassHit(o, E, root, BassRole.CHORD_TONE, accent(o)) for o in eighths]
    if style == "root_quarters":
        return [BassHit(o, Q, root, BassRole.CHORD_TONE, accent(o)) for o in quarters]
    if style == "root_fifth":
        cycle = (root, fifth, octave, fifth)
        return [
            BassHit(o, Q, cycle[i % 4], BassRole.CHORD_TONE, accent(o))
            for i, o in enumerate(quarters)
        ]
    if style == "follow_riff":
        return [
            BassHit(h.offset, h.duration, root, BassRole.CHORD_TONE, accent(h.offset)) for h in riff
        ]
    if style == "approach":
        hits = [BassHit(o, E, root, BassRole.CHORD_TONE, accent(o)) for o in eighths]
        if approach is not None and len(hits) > 1:
            last = hits[-1]
            hits[-1] = BassHit(last.offset, E, approach, BassRole.CHROMATIC_APPROACH)
        return hits
    if style == "neighbor":
        return [
            BassHit(o, E, neighbor, BassRole.NEIGHBOR)
            if i % 4 == 2
            else BassHit(o, E, root, BassRole.CHORD_TONE, accent(o))
            for i, o in enumerate(eighths)
        ]
    if style == "walking":
        third = span.quality.third
        if third is None:
            raise ValueError("a walking line needs a chord with a third")
        closing: tuple[int, BassRole] = (
            (approach, BassRole.CHROMATIC_APPROACH)
            if approach is not None
            else (fifth, BassRole.CHORD_TONE)
        )
        n = len(quarters)
        line: list[tuple[int, BassRole]]
        if n >= 4:
            line = [
                (root, BassRole.CHORD_TONE),
                (root + 2, BassRole.PASSING),
                (root + third, BassRole.CHORD_TONE),
            ]
            line += [(fifth, BassRole.CHORD_TONE)] * (n - 4) + [closing]
        elif n == 3:
            line = [(root, BassRole.CHORD_TONE), (root + third, BassRole.CHORD_TONE), closing]
        else:
            line = [(root, BassRole.CHORD_TONE), closing][:n]
        return [
            BassHit(o, Q, pitch, role, accent(o))
            for o, (pitch, role) in zip(quarters, line, strict=True)
        ]
    if style == "blues_boogie":
        intervals, roles = _BOOGIE_BLUE if blue else _BOOGIE
        return [
            BassHit(o, E, root + intervals[i % 8], _ROLE_CODES[roles[i % 8]], accent(o))
            for i, o in enumerate(eighths)
        ]
    raise ValueError(f"unknown bass style: {style}")


# ------------------------------------------------------------------------- drums


@dataclass(frozen=True, slots=True)
class Groove:
    """A one-bar drum pattern on a grid of ``steps`` cells of ``step_ticks`` each."""

    name: str
    steps: int
    step_ticks: int
    pattern: Mapping[str, tuple[int, ...]]
    ghosts: tuple[int, ...] = ()
    fill: tuple[str, ...] = field(default=())


_FILL_8: Final = (
    "snare",
    "snare",
    "tom_high",
    "tom_high",
    "tom_mid",
    "tom_mid",
    "tom_floor",
    "tom_floor",
)
_FILL_6: Final = ("snare", "snare", "tom_high", "tom_mid", "tom_floor", "tom_floor")
_HH8: Final = (0, 2, 4, 6, 8, 10, 12, 14)

GROOVES: Final[dict[str, tuple[Groove, ...]]] = {
    "4/4": (
        Groove(
            "rock", 16, S, {"kick": (0, 8, 10), "snare": (4, 12), "hh_closed": _HH8}, (), _FILL_8
        ),
        Groove(
            "rock_ghosts",
            16,
            S,
            {"kick": (0, 6, 8), "snare": (4, 12), "hh_closed": _HH8},
            (7, 15),
            _FILL_8,
        ),
        Groove(
            "half_time", 16, S, {"kick": (0, 3, 10), "snare": (8,), "hh_closed": _HH8}, (), _FILL_8
        ),
        Groove(
            "cross_stick",
            16,
            S,
            {"kick": (0, 8, 10), "rim": (4, 12), "hh_closed": _HH8},
            (),
            _FILL_8,
        ),
        Groove(
            "double_time",
            16,
            S,
            {"kick": (0, 4, 8, 12), "snare": (2, 6, 10, 14), "hh_closed": (0, 4, 8, 12)},
            (),
            _FILL_8,
        ),
        Groove(
            "four_floor",
            16,
            S,
            {
                "kick": (0, 4, 8, 12),
                "snare": (4, 12),
                "hh_closed": (0, 4, 8, 12),
                "hh_open": (2, 6, 10, 14),
            },
            (),
            _FILL_8,
        ),
        Groove(
            "ride",
            16,
            S,
            {
                "kick": (0, 8, 11),
                "snare": (4, 12),
                "ride": (0, 2, 6, 10, 14),
                "ride_bell": (4, 8, 12),
                "hh_pedal": (4, 12),
            },
            (),
            _FILL_8,
        ),
    ),
    "3/4": (
        Groove(
            "waltz",
            12,
            S,
            {"kick": (0,), "snare": (4, 8), "hh_closed": (0, 2, 4, 6, 8, 10)},
            (),
            _FILL_6,
        ),
        Groove(
            "waltz_ride",
            12,
            S,
            {"kick": (0, 6), "snare": (4, 8), "ride": (2, 4, 6, 8, 10), "hh_pedal": (4, 8)},
            (),
            _FILL_6,
        ),
    ),
    "shuffle": (
        Groove(
            "shuffle",
            12,
            T8,
            {"kick": (0, 6), "snare": (3, 9), "hh_closed": (0, 2, 3, 5, 6, 8, 9, 11)},
            (1, 7),
            _FILL_6,
        ),
        Groove(
            "shuffle_ride",
            12,
            T8,
            {"kick": (0, 6, 8), "snare": (3, 9), "ride": (0, 2, 3, 5, 6, 8, 9, 11)},
            (),
            _FILL_6,
        ),
    ),
}

_TIMEKEEPERS: Final = frozenset({"hh_closed", "hh_open", "ride", "ride_bell"})


@dataclass(frozen=True, slots=True)
class DrumHit:
    """A drum stroke on a class, at a tick inside the bar."""

    offset: int
    drum_class: str
    dynamic: Dynamic
    duration: int


def drum_bar(
    groove: Groove,
    *,
    cymbal: str | None = None,
    fill: bool = False,
    open_hat: bool = False,
    backbeat: Dynamic = Dynamic.NORMAL,
) -> list[DrumHit]:
    """One bar of a groove, optionally opened by a cymbal and closed by a tom fill."""
    cells: dict[tuple[int, str], Dynamic] = {}
    for drum_class, steps in groove.pattern.items():
        for step in steps:
            dynamic = backbeat if drum_class == "snare" else Dynamic.NORMAL
            cells[(step, drum_class)] = dynamic
    for step in groove.ghosts:
        cells.setdefault((step, "snare"), Dynamic.GHOST)
    if open_hat:
        closed = sorted(step for step, cls in cells if cls == "hh_closed")
        if closed:
            del cells[(closed[-1], "hh_closed")]
            cells[(closed[-1], "hh_open")] = Dynamic.NORMAL
    if cymbal is not None:
        for key in [k for k in cells if k[0] == 0 and k[1] in _TIMEKEEPERS]:
            del cells[key]
        cells[(0, cymbal)] = Dynamic.ACCENT
        cells[(0, "kick")] = Dynamic.ACCENT
    if fill and groove.fill:
        start = groove.steps - len(groove.fill)
        cells = {k: v for k, v in cells.items() if k[0] < start}
        for i, drum_class in enumerate(groove.fill):
            last = i == len(groove.fill) - 1
            cells[(start + i, drum_class)] = Dynamic.ACCENT if last else Dynamic.NORMAL
        cells[(start, "kick")] = Dynamic.NORMAL
    return [
        DrumHit(step * groove.step_ticks, cls, dynamic, groove.step_ticks)
        for (step, cls), dynamic in sorted(cells.items())
    ]
