# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Invariant representations of a bass line and a drum part.

Bass: each note becomes an event (interval from the previous note, ratio of inter-onset
intervals, coarse ratio of durations). Intervals ignore transposition; ratios ignore
tempo, including half- and double-time feels. A second form uses the scale degree in the
key, for "same line, other key" evidence.

Drums: each beat becomes a code of onsets on 4 sub-steps for 5 coarse classes (kick,
snare, hats, toms, cymbals). Stretched (x2) and compressed (x0.5) versions let a half-time
or double-time copy match too.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from backline.config import DrumsConfig
from backline.score import Note
from backline.theory.grid import Beat

MAX_INTERVAL: Final = 24
IOI_BINS: Final = 3
"""Inter-onset ratios are binned in thirds of an octave (log2), clipped to +-3 octaves."""
SUB_STEPS: Final = 4


@dataclass(frozen=True, slots=True)
class EventSequence:
    """Event codes and, for each event, the tick span of the note it ends on."""

    codes: npt.NDArray[np.int64]
    starts: list[int]
    ends: list[int]


def monophonic(notes: Sequence[Note]) -> list[Note]:
    """Lowest note of every onset (double stops count as their root)."""
    lowest: dict[int, Note] = {}
    for note in notes:
        current = lowest.get(note.start)
        if current is None or note.pitch < current.pitch:
            lowest[note.start] = note
    return [lowest[t] for t in sorted(lowest)]


def _ratio_bin(new: int, old: int) -> int:
    if new <= 0 or old <= 0:
        return 0
    return int(max(-3 * IOI_BINS, min(3 * IOI_BINS, round(math.log2(new / old) * IOI_BINS))))


def _duration_bin(new: int, old: int) -> int:
    if old <= 0:
        return 0
    ratio = new / old
    return 0 if ratio < 0.7 else 2 if ratio > 1.4 else 1


def bass_events(notes: Sequence[Note]) -> EventSequence:
    """Interval/rhythm events; the first two notes only provide context."""
    line = monophonic(notes)
    codes, starts, ends = [], [], []
    for i in range(2, len(line)):
        interval = max(-MAX_INTERVAL, min(MAX_INTERVAL, line[i].pitch - line[i - 1].pitch))
        ioi = _ratio_bin(line[i].start - line[i - 1].start, line[i - 1].start - line[i - 2].start)
        dur = _duration_bin(line[i].duration, line[i - 1].duration)
        codes.append(((interval + MAX_INTERVAL) * 32 + (ioi + 16)) * 4 + dur)
        starts.append(line[i - 1].start)
        ends.append(line[i].end)
    return EventSequence(np.array(codes, dtype=np.int64), starts, ends)


def degree_events(notes: Sequence[Note], tonic_at: Callable[[int], int]) -> EventSequence:
    """Scale degree of each note with the rhythm ratio leading to it."""
    line = monophonic(notes)
    codes, starts, ends = [], [], []
    for i in range(2, len(line)):
        degree = (line[i].pitch - tonic_at(line[i].start)) % 12
        ioi = _ratio_bin(line[i].start - line[i - 1].start, line[i - 1].start - line[i - 2].start)
        codes.append(degree * 32 + ioi + 16)
        starts.append(line[i - 1].start)
        ends.append(line[i].end)
    return EventSequence(np.array(codes, dtype=np.int64), starts, ends)


# ------------------------------------------------------------------- drums


@dataclass(frozen=True, slots=True)
class DrumSequence:
    """One code per beat; ``beats`` keeps the beat grid for mapping back to time."""

    codes: npt.NDArray[np.int64]
    beats: list[Beat]


def drum_cells(
    notes: Sequence[Note], beats: Sequence[Beat], drums: DrumsConfig
) -> npt.NDArray[np.int64]:
    """Per beat, a bitmask of (sub-step, coarse class) onsets."""
    groups = list(drums.coarse)
    cells = np.zeros(len(beats), dtype=np.int64)
    starts = [b.start for b in beats]
    for note in notes:
        drum_class = drums.class_of(note.pitch)
        if drum_class is None:
            continue
        index = bisect.bisect_right(starts, note.start) - 1
        if index < 0:
            continue
        beat = beats[index]
        if note.start >= beat.end:
            continue
        sub = min((note.start - beat.start) * SUB_STEPS // (beat.end - beat.start), SUB_STEPS - 1)
        group = groups.index(drums.coarse_of(drum_class))
        cells[index] |= 1 << (sub * len(groups) + group)
    return cells


def drum_sequence(notes: Sequence[Note], beats: Sequence[Beat], drums: DrumsConfig) -> DrumSequence:
    return DrumSequence(drum_cells(notes, beats, drums), list(beats))


def rescale(cells: npt.NDArray[np.int64], factor: float, n_groups: int) -> npt.NDArray[np.int64]:
    """Stretch (factor 2: half-time) or compress (0.5: double-time) a beat-code sequence."""
    total = len(cells) * SUB_STEPS
    onsets = [
        (beat * SUB_STEPS + sub, group)
        for beat, code in enumerate(cells.tolist())
        for sub in range(SUB_STEPS)
        for group in range(n_groups)
        if code >> (sub * n_groups + group) & 1
    ]
    length = max(int(len(cells) * factor), 1)
    out = np.zeros(length, dtype=np.int64)
    for step, group in onsets:
        moved = int(step * factor)
        if moved >= min(total * factor, length * SUB_STEPS):
            continue
        beat, sub = divmod(moved, SUB_STEPS)
        out[beat] |= 1 << (sub * n_groups + group)
    return out


def beat_jaccard(a: int, b: int) -> float:
    """Jaccard similarity of two beat codes (empty vs empty counts as identical)."""
    union = bin(a | b).count("1")
    return 1.0 if union == 0 else bin(a & b).count("1") / union
