# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Guitar Pro 5 writer built on PyGuitarPro.

Each track goes into one voice (see :mod:`backline.formats.tablature`); only the initial
tempo is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import guitarpro as gp

from backline.formats.tablature import (
    DEFAULT_TUNINGS,
    PERCUSSION_STRINGS,
    NoteValue,
    Placed,
    TabLayoutError,
    TabNote,
    layout_bar,
    place_fretted,
    place_percussion,
)
from backline.score import TPQ, NoteFlag, Score

MAX_STRINGS: Final = 7


class GP5WriteError(TabLayoutError):
    """Raised when a score cannot be represented in Guitar Pro 5."""


# ------------------------------------------------------------------------- writing


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
        placed: list[Placed]
        if track.is_percussion:
            channel = gp.MidiChannel(channel=9, effectChannel=9, instrument=0)
            strings = [gp.GuitarString(n, 0) for n in range(1, PERCUSSION_STRINGS + 1)]
            placed = place_percussion(track.notes, starts[-1])
        else:
            main, effect = next(channels)
            channel = gp.MidiChannel(channel=main, effectChannel=effect, instrument=track.program)
            tuning = track.tuning or DEFAULT_TUNINGS[track.role]
            if len(tuning) > MAX_STRINGS:
                raise GP5WriteError(f"track {track.name!r} has more than {MAX_STRINGS} strings")
            strings = [gp.GuitarString(i + 1, p) for i, p in enumerate(reversed(tuning))]
            placed = place_fretted(track.notes, tuning, frets=24)
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
        for index, measure in enumerate(gp_track.measures):
            voice = measure.voices[0]
            for tab_beat in layout_bar(placed, starts[index], starts[index + 1]):
                beat = gp.Beat(
                    voice,
                    duration=_gp_duration(tab_beat.value),
                    status=gp.BeatStatus.normal if tab_beat.notes else gp.BeatStatus.rest,
                )
                beat.notes.extend(_gp_note(beat, n) for n in tab_beat.notes)
                voice.beats.append(beat)
            measure.voices[1].beats.append(gp.Beat(measure.voices[1], status=gp.BeatStatus.empty))
    return song


def _gp_duration(value: NoteValue) -> gp.Duration:
    tuplet = gp.Tuplet(3, 2) if value.triplet else gp.Tuplet(1, 1)
    return gp.Duration(value=value.value, isDotted=value.dotted, tuplet=tuplet)


def _gp_velocity(velocity: int) -> int:
    """Nearest Guitar Pro dynamic (ppp..fff) so the value survives a round trip."""
    step = gp.Velocities.velocityIncrement
    level = min(max(round((velocity - gp.Velocities.minVelocity) / step), 0), 7)
    return int(gp.Velocities.minVelocity + level * step)


def _gp_note(beat: gp.Beat, tab_note: TabNote) -> gp.Note:
    note = tab_note.placed.note
    tied = tab_note.tie_in
    effect = gp.NoteEffect(
        palmMute=bool(note.flags & NoteFlag.PALM_MUTE),
        ghostNote=not tied and bool(note.flags & NoteFlag.GHOST),
        accentuatedNote=not tied and bool(note.flags & NoteFlag.ACCENT),
        heavyAccentuatedNote=not tied and bool(note.flags & NoteFlag.HEAVY_ACCENT),
    )
    kind: Any
    if tied:
        kind = gp.NoteType.tie
    elif note.flags & NoteFlag.DEAD:
        kind = gp.NoteType.dead
    else:
        kind = gp.NoteType.normal
    return gp.Note(
        beat,
        value=tab_note.placed.fret,
        velocity=_gp_velocity(note.velocity),
        string=tab_note.placed.string,
        effect=effect,
        type=kind,
    )
