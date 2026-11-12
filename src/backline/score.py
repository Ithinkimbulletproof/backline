# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical in-memory song representation shared by readers, writers and analysis.

Time is measured in ticks at a fixed resolution (:data:`TPQ` ticks per quarter note),
never in seconds, so tempo changes do not move notes relative to the bar grid.
Drum notes keep the General MIDI percussion number they were read with; mapping to
drum classes happens later, driven by the config.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from enum import IntFlag, StrEnum
from typing import Final

TPQ: Final = 960
"""Ticks per quarter note. 960 = 2^6 * 3 * 5 divides evenly into 64ths and triplets."""


class TrackRole(StrEnum):
    """What a track does in the band."""

    GUITAR = "guitar"
    BASS = "bass"
    DRUMS = "drums"
    OTHER = "other"


class NoteFlag(IntFlag):
    """Articulations and accents that survive conversion between formats."""

    NONE = 0
    GHOST = 1
    ACCENT = 2
    HEAVY_ACCENT = 4
    PALM_MUTE = 8
    DEAD = 16


@dataclass(frozen=True, slots=True, order=True)
class Note:
    """A single note. Ordering is by start tick, then pitch."""

    start: int
    pitch: int
    duration: int
    velocity: int = 90
    flags: NoteFlag = NoteFlag.NONE

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"note start must be >= 0, got {self.start}")
        if self.duration <= 0:
            raise ValueError(f"note duration must be > 0, got {self.duration}")
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"note pitch must be within 0..127, got {self.pitch}")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"note velocity must be within 1..127, got {self.velocity}")

    @property
    def end(self) -> int:
        """Tick at which the note stops sounding."""
        return self.start + self.duration

    def transposed(self, semitones: int) -> Note:
        """Return a copy shifted by ``semitones``."""
        return replace(self, pitch=self.pitch + semitones)


@dataclass(frozen=True, slots=True)
class TimeSignature:
    """A meter that starts at bar ``bar`` (0-based) and holds until the next change."""

    bar: int
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.bar < 0:
            raise ValueError("time signature bar must be >= 0")
        if self.numerator <= 0:
            raise ValueError("time signature numerator must be > 0")
        if self.denominator <= 0 or self.denominator & (self.denominator - 1):
            raise ValueError("time signature denominator must be a power of two")

    def bar_ticks(self, tpq: int = TPQ) -> int:
        """Length of one bar in ticks."""
        return self.numerator * self.beat_ticks(tpq)

    def beat_ticks(self, tpq: int = TPQ) -> int:
        """Length of the notated beat unit (the denominator) in ticks."""
        return tpq * 4 // self.denominator

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True, slots=True)
class Tempo:
    """A tempo in quarter notes per minute that starts at ``tick``."""

    tick: int
    bpm: float

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("tempo tick must be >= 0")
        if self.bpm <= 0:
            raise ValueError("tempo must be > 0 bpm")


@dataclass(slots=True)
class Track:
    """One instrument part."""

    name: str
    role: TrackRole
    notes: list[Note] = field(default_factory=list)
    program: int = 0
    is_percussion: bool = False
    tuning: tuple[int, ...] | None = None
    """Open-string pitches from the lowest string up, for fretted instruments."""
    instrument: str = ""
    """Instrument hint from the source file, e.g. Guitar Pro's ``electricBass``."""
    percussion_names: dict[int, str] = field(default_factory=dict)
    """Names the source file gives to percussion notes, e.g. ``{46: "Hi-Hat.Open"}``."""

    def sorted_notes(self) -> list[Note]:
        """Notes ordered by start tick, then pitch."""
        return sorted(self.notes)


@dataclass(slots=True)
class Score:
    """A whole song: meter and tempo maps plus its tracks."""

    n_bars: int
    time_signatures: list[TimeSignature]
    tempos: list[Tempo]
    tracks: list[Track] = field(default_factory=list)
    tpq: int = TPQ
    title: str = ""
    artist: str = ""

    def __post_init__(self) -> None:
        if self.n_bars <= 0:
            raise ValueError("a score needs at least one bar")
        if not self.time_signatures or self.time_signatures[0].bar != 0:
            raise ValueError("the first time signature must start at bar 0")
        if not self.tempos or self.tempos[0].tick != 0:
            raise ValueError("the first tempo must start at tick 0")
        bars = [ts.bar for ts in self.time_signatures]
        if bars != sorted(set(bars)):
            raise ValueError("time signatures must be sorted by bar without duplicates")
        ticks = [t.tick for t in self.tempos]
        if ticks != sorted(set(ticks)):
            raise ValueError("tempos must be sorted by tick without duplicates")

    def time_signature_at(self, bar: int) -> TimeSignature:
        """Meter in effect at a 0-based bar index."""
        index = bisect.bisect_right([ts.bar for ts in self.time_signatures], bar) - 1
        return self.time_signatures[max(index, 0)]

    def bar_starts(self) -> list[int]:
        """Start tick of every bar, plus the end tick of the song as the last element."""
        starts = [0]
        for bar in range(self.n_bars):
            starts.append(starts[-1] + self.time_signature_at(bar).bar_ticks(self.tpq))
        return starts

    @property
    def end(self) -> int:
        """Total length in ticks."""
        return self.bar_starts()[-1]

    def bar_of(self, tick: int) -> int:
        """0-based index of the bar containing ``tick``."""
        starts = self.bar_starts()
        if not 0 <= tick < starts[-1]:
            raise ValueError(f"tick {tick} is outside the song (0..{starts[-1] - 1})")
        return bisect.bisect_right(starts, tick) - 1

    def tempo_at(self, tick: int) -> float:
        """Tempo in bpm in effect at ``tick``."""
        index = bisect.bisect_right([t.tick for t in self.tempos], tick) - 1
        return self.tempos[max(index, 0)].bpm

    def tracks_with_role(self, role: TrackRole) -> list[Track]:
        """All tracks that play ``role``."""
        return [track for track in self.tracks if track.role is role]
