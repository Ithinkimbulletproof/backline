# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Guitar Pro 5 writer built on PyGuitarPro.

Used to write synthetic tabs for tests and, later, to export generated parts as tabs.
Each track is written into a single voice: note boundaries become beat boundaries,
held notes continue as tied notes, and every beat length is split into durations Guitar
Pro can notate. Boundaries are snapped to a grid of 1/24 of a quarter note.
Only the initial tempo is written.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import guitarpro as gp

from backline.score import TPQ, Note, NoteFlag, Score, Track, TrackRole

GRID: Final = TPQ // 24
"""Smallest boundary step: 40 ticks covers 32nds, 16th triplets and 64th triplets."""

MAX_STRINGS: Final = 7
DEFAULT_TUNINGS: Final = {
    TrackRole.GUITAR: (40, 45, 50, 55, 59, 64),
    TrackRole.BASS: (28, 33, 38, 43),
    TrackRole.OTHER: (40, 45, 50, 55, 59, 64),
}
PERCUSSION_STRINGS: Final = 6


class GP5WriteError(ValueError):
    """Raised when a score cannot be represented in Guitar Pro 5."""


@dataclass(frozen=True, slots=True)
class _NoteValue:
    ticks: int
    value: int
    dotted: bool
    triplet: bool
    cost: float


def _duration_table() -> tuple[_NoteValue, ...]:
    table = []
    for value in (1, 2, 4, 8, 16, 32, 64):
        plain = TPQ * 4 // value
        table.append(_NoteValue(plain, value, False, False, 1.0))
        table.append(_NoteValue(plain * 3 // 2, value, True, False, 1.1))
        table.append(_NoteValue(plain * 2 // 3, value, False, True, 1.3))
    return tuple(sorted((d for d in table if d.ticks % GRID == 0), key=lambda d: -d.ticks))


_DURATIONS: Final = _duration_table()


@lru_cache(maxsize=4096)
def split_ticks(ticks: int) -> tuple[_NoteValue, ...]:
    """Fewest notatable durations that add up to ``ticks`` (a multiple of :data:`GRID`)."""
    if ticks <= 0 or ticks % GRID:
        raise GP5WriteError(f"cannot notate a length of {ticks} ticks")
    best: dict[int, tuple[float, tuple[_NoteValue, ...]]] = {0: (0.0, ())}
    for total in range(GRID, ticks + 1, GRID):
        options = [
            (best[total - d.ticks][0] + d.cost, (*best[total - d.ticks][1], d))
            for d in _DURATIONS
            if d.ticks <= total and total - d.ticks in best
        ]
        if options:
            best[total] = min(options, key=lambda o: o[0])
    if ticks not in best:
        raise GP5WriteError(f"cannot notate a length of {ticks} ticks")
    return tuple(sorted(best[ticks][1], key=lambda d: -d.ticks))


def _gp_duration(d: _NoteValue) -> gp.Duration:
    tuplet = gp.Tuplet(3, 2) if d.triplet else gp.Tuplet(1, 1)
    return gp.Duration(value=d.value, isDotted=d.dotted, tuplet=tuplet)


def _snap(tick: int) -> int:
    return round(tick / GRID) * GRID


def _gp_velocity(velocity: int) -> int:
    """Nearest Guitar Pro dynamic (ppp..fff) so the value survives a round trip."""
    step = gp.Velocities.velocityIncrement
    level = min(max(round((velocity - gp.Velocities.minVelocity) / step), 0), 7)
    return int(gp.Velocities.minVelocity + level * step)


@dataclass(slots=True)
class _Placed:
    start: int
    end: int
    note: Note
    string: int
    fret: int


def write_gp5(score: Score, path: Path) -> None:
    """Write ``score`` to a ``.gp5`` file."""
    gp.write(score_to_gp5(score), str(path), version=(5, 1, 0))


def score_to_gp5(score: Score) -> gp.Song:
    """Build a PyGuitarPro song from a score."""
    if score.tpq != TPQ:
        raise GP5WriteError(f"scores must use {TPQ} ticks per quarter, got {score.tpq}")
    song = gp.Song()
    song.title = score.title
    song.artist = score.artist
    song.tempo = round(score.tempos[0].bpm)
    song.measureHeaders = []
    song.tracks = []

    starts = score.bar_starts()
    for bar in range(score.n_bars):
        ts = score.time_signature_at(bar)
        header = gp.MeasureHeader(
            number=bar + 1,
            start=gp.Duration.quarterTime + starts[bar],
            timeSignature=gp.TimeSignature(
                numerator=ts.numerator, denominator=gp.Duration(value=ts.denominator)
            ),
        )
        song.addMeasureHeader(header)

    channels = iter((c, c + 1) for c in range(0, 16, 2) if 9 not in (c, c + 1))
    for number, track in enumerate(score.tracks, start=1):
        if track.is_percussion:
            channel = gp.MidiChannel(channel=9, effectChannel=9, instrument=0)
            strings = [gp.GuitarString(n, 0) for n in range(1, PERCUSSION_STRINGS + 1)]
            placed = _place_percussion(track, starts[-1])
        else:
            main, effect = next(channels)
            channel = gp.MidiChannel(channel=main, effectChannel=effect, instrument=track.program)
            tuning = track.tuning or DEFAULT_TUNINGS[track.role]
            if len(tuning) > MAX_STRINGS:
                raise GP5WriteError(f"track {track.name!r} has more than {MAX_STRINGS} strings")
            strings = [gp.GuitarString(i + 1, p) for i, p in enumerate(reversed(tuning))]
            placed = _place_fretted(track, tuning, frets=24)
        gp_track = gp.Track(
            song,
            number=number,
            name=track.name,
            isPercussionTrack=track.is_percussion,
            strings=strings,
            channel=channel,
            fretCount=24,
        )
        song.tracks.append(gp_track)
        _fill_measures(gp_track, placed, starts)
    return song


def _place_fretted(track: Track, tuning: tuple[int, ...], *, frets: int) -> list[_Placed]:
    """Give every note a string and fret; notes that cannot be fingered are dropped."""
    n_strings = len(tuning)
    placed: list[_Placed] = []
    sounding: list[_Placed] = []
    notes = sorted(track.notes)
    i = 0
    while i < len(notes):
        onset = _snap(notes[i].start)
        group = []
        while i < len(notes) and _snap(notes[i].start) == onset:
            group.append(notes[i])
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
            ordered = [o for o in options if o[1] >= lowest_free] or options
            fret, idx = min(ordered)
            string = n_strings - idx
            busy.add(string)
            lowest_free = idx + 1
            end = max(_snap(note.end), onset + GRID)
            item = _Placed(onset, end, note, string, fret)
            placed.append(item)
            sounding.append(item)
    return placed


def _place_percussion(track: Track, song_end: int) -> list[_Placed]:
    """Drum hits sound until the next hit, as they are notated in tabs."""
    onsets = sorted({_snap(n.start) for n in track.notes})
    next_onset = dict(zip(onsets, [*onsets[1:], song_end], strict=True))
    placed = []
    by_onset: dict[int, int] = {}
    for note in sorted(track.notes):
        onset = _snap(note.start)
        string = by_onset.get(onset, 0) + 1
        if string > PERCUSSION_STRINGS:
            continue
        by_onset[onset] = string
        end = max(next_onset[onset], onset + GRID)
        placed.append(_Placed(onset, end, note, string, note.pitch))
    return placed


def _fill_measures(gp_track: gp.Track, placed: list[_Placed], starts: list[int]) -> None:
    for index, measure in enumerate(gp_track.measures):
        bar_start, bar_end = starts[index], starts[index + 1]
        voice = measure.voices[0]
        inside = [p for p in placed if p.start < bar_end and p.end > bar_start]
        bounds = {bar_start, bar_end}
        for p in inside:
            bounds.update(t for t in (p.start, p.end) if bar_start < t < bar_end)
        edges = sorted(bounds)
        for seg_start, seg_end in itertools.pairwise(edges):
            active = [p for p in inside if p.start <= seg_start < p.end]
            for k, piece in enumerate(split_ticks(seg_end - seg_start)):
                beat = gp.Beat(
                    voice,
                    duration=_gp_duration(piece),
                    status=gp.BeatStatus.normal if active else gp.BeatStatus.rest,
                )
                for p in active:
                    tied = p.start < seg_start or k > 0
                    beat.notes.append(_gp_note(beat, p, tied=tied))
                voice.beats.append(beat)
        measure.voices[1].beats.append(gp.Beat(measure.voices[1], status=gp.BeatStatus.empty))


def _gp_note(beat: gp.Beat, placed: _Placed, *, tied: bool) -> gp.Note:
    note = placed.note
    effect = gp.NoteEffect(
        palmMute=bool(note.flags & NoteFlag.PALM_MUTE),
        ghostNote=not tied and bool(note.flags & NoteFlag.GHOST),
        accentuatedNote=not tied and bool(note.flags & NoteFlag.ACCENT),
        heavyAccentuatedNote=not tied and bool(note.flags & NoteFlag.HEAVY_ACCENT),
    )
    if tied:
        kind = gp.NoteType.tie
    elif note.flags & NoteFlag.DEAD:
        kind = gp.NoteType.dead
    else:
        kind = gp.NoteType.normal
    return gp.Note(
        beat,
        value=placed.fret,
        velocity=_gp_velocity(note.velocity),
        string=placed.string,
        effect=effect,
        type=kind,
    )
