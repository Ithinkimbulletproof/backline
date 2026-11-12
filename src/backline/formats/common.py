# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Pieces shared by the readers: errors, read statistics and tie-aware note collection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from dataclasses import replace

from backline.score import Note, TrackRole

ReadStats = Counter[str]
"""Counters of things a reader skipped or repaired, e.g. ``grace_notes_dropped``."""

TIE_TOLERANCE = 2
"""Ticks of rounding slack allowed between a tied note's end and its continuation."""


class ScoreReadError(ValueError):
    """Raised when a file cannot be read as a score."""


class UnsupportedFormatError(ScoreReadError):
    """Raised for file formats the readers do not handle (yet)."""


def guess_role(program: int, percussion: bool) -> TrackRole:
    """First guess of a track's role from its General MIDI program; ingest refines it."""
    if percussion:
        return TrackRole.DRUMS
    if 32 <= program <= 39:
        return TrackRole.BASS
    if 24 <= program <= 31:
        return TrackRole.GUITAR
    return TrackRole.OTHER


class TrackNotes:
    """Collects the notes of one track and merges tied continuations into their origin."""

    def __init__(self) -> None:
        self.notes: list[Note] = []
        self._last: dict[Hashable, int] = {}

    def add(self, key: Hashable, note: Note) -> None:
        """Add a new note; ``key`` identifies what a later tie may continue (e.g. a string)."""
        self._last[key] = len(self.notes)
        self.notes.append(note)

    def extend(self, key: Hashable, start: int, end: int) -> bool:
        """Lengthen the last note under ``key`` to ``end`` if it stops at ``start``."""
        index = self._last.get(key)
        if index is None:
            return False
        note = self.notes[index]
        if abs(note.end - start) > TIE_TOLERANCE:
            return False
        self.notes[index] = replace(note, duration=max(end - note.start, 1))
        return True
