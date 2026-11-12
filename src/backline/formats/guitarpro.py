# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Guitar Pro 3/4/5 files via PyGuitarPro: read ``.gp3``/``.gp4``/``.gp5``, write ``.gp5``.

Reading keeps every voice, merges tied notes, drops grace notes (counted in the stats),
applies the triplet feel of each bar and adds the capo to fretted pitches. Repeats are not
unrolled: the score follows the notation order. Writing puts each track into one voice
(see :mod:`backline.formats.tablature`) and writes only the initial tempo.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Final

import guitarpro as gp

from backline.formats.common import ReadStats, ScoreReadError, TrackNotes, guess_role
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
from backline.score import TPQ, Note, NoteFlag, Score, Tempo, TimeSignature, Track
from backline.timing import Feel, apply_feel

MAX_STRINGS: Final = 7
_FEELS: Final = {
    gp.TripletFeel.none: Feel.STRAIGHT,
    gp.TripletFeel.eighth: Feel.TRIPLET_8,
    gp.TripletFeel.sixteenth: Feel.TRIPLET_16,
}


class GP5WriteError(TabLayoutError):
    """Raised when a score cannot be represented in Guitar Pro 5."""


# ------------------------------------------------------------------------- reading


def read_guitarpro(
    path: Path, *, stats: ReadStats | None = None, apply_triplet_feel: bool = True
) -> Score:
    """Read a ``.gp3``, ``.gp4`` or ``.gp5`` file."""
    try:
        song = gp.parse(str(path))
    except (
        gp.GPException,
        struct.error,
        UnicodeDecodeError,
        EOFError,
        ValueError,
        IndexError,
        KeyError,
        AttributeError,
    ) as exc:
        raise ScoreReadError(f"cannot parse {path.name}: {exc}") from exc
    return song_to_score(song, stats=stats, apply_triplet_feel=apply_triplet_feel)


def song_to_score(
    song: gp.Song, *, stats: ReadStats | None = None, apply_triplet_feel: bool = True
) -> Score:
    """Convert a parsed PyGuitarPro song to a score."""
    counters = stats if stats is not None else ReadStats()
    headers = song.measureHeaders
    if not headers:
        raise ScoreReadError("the file has no bars")
    origin = headers[0].start

    time_signatures: list[TimeSignature] = []
    for bar, header in enumerate(headers):
        meter = (header.timeSignature.numerator, header.timeSignature.denominator.value)
        if not time_signatures or meter != (
            time_signatures[-1].numerator,
            time_signatures[-1].denominator,
        ):
            time_signatures.append(TimeSignature(bar, *meter))

    tempos: dict[int, float] = {0: float(song.tempo)}
    tracks = [_read_track(t, origin, tempos, counters) for t in song.tracks]
    score = Score(
        n_bars=len(headers),
        time_signatures=time_signatures,
        tempos=[Tempo(tick, bpm) for tick, bpm in sorted(tempos.items())],
        tracks=tracks,
        title=song.title or "",
        artist=song.artist or "",
    )
    if apply_triplet_feel:
        feels = {bar: _FEELS.get(h.tripletFeel, Feel.STRAIGHT) for bar, h in enumerate(headers)}
        apply_feel(score, feels)
    return score


def _read_track(track: gp.Track, origin: int, tempos: dict[int, float], stats: ReadStats) -> Track:
    percussion = bool(track.isPercussionTrack or track.channel.isPercussionChannel)
    strings = {s.number: s.value for s in track.strings}
    tuning = None if percussion else tuple(strings[n] for n in sorted(strings, reverse=True))
    capo = 0 if percussion else int(track.offset)
    collected = TrackNotes()
    for measure in track.measures:
        for voice_index, voice in enumerate(measure.voices):
            for beat in voice.beats:
                _read_tempo_change(beat, origin, tempos)
                if beat.status != gp.BeatStatus.normal or beat.start is None:
                    continue
                start = beat.start - origin
                end = start + round(beat.duration.time)
                for gp_note in beat.notes:
                    _read_note(gp_note, voice_index, start, end, capo, percussion, collected, stats)
    return Track(
        name=track.name,
        role=guess_role(int(track.channel.instrument), percussion),
        notes=sorted(collected.notes),
        program=int(track.channel.instrument),
        is_percussion=percussion,
        tuning=tuning,
    )


def _read_tempo_change(beat: gp.Beat, origin: int, tempos: dict[int, float]) -> None:
    change = beat.effect.mixTableChange
    if change is None or change.tempo is None or beat.start is None:
        return
    value = change.tempo.value
    if value and value > 0:
        tempos.setdefault(beat.start - origin, float(value))


def _read_note(
    gp_note: gp.Note,
    voice: int,
    start: int,
    end: int,
    capo: int,
    percussion: bool,
    collected: TrackNotes,
    stats: ReadStats,
) -> None:
    if gp_note.type == gp.NoteType.rest:
        return
    pitch = gp_note.value if percussion else gp_note.realValue + capo
    key = (voice, gp_note.string)
    if gp_note.type == gp.NoteType.tie:
        if not collected.extend(key, start, end):
            stats["orphan_ties"] += 1
        return
    if gp_note.effect.isGrace:
        stats["grace_notes_dropped"] += 1
    if not 0 <= pitch <= 127:
        stats["notes_out_of_midi_range"] += 1
        return
    effect = gp_note.effect
    flags = NoteFlag.NONE
    if effect.ghostNote:
        flags |= NoteFlag.GHOST
    if effect.accentuatedNote:
        flags |= NoteFlag.ACCENT
    if effect.heavyAccentuatedNote:
        flags |= NoteFlag.HEAVY_ACCENT
    if effect.palmMute:
        flags |= NoteFlag.PALM_MUTE
    if gp_note.type == gp.NoteType.dead:
        flags |= NoteFlag.DEAD
    velocity = min(max(int(gp_note.velocity), 1), 127)
    collected.add(key, Note(start, pitch, max(end - start, 1), velocity, flags))


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
