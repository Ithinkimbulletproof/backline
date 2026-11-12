# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Standard MIDI files <-> :class:`~backline.score.Score`.

MIDI carries no articulations, so note flags are lost on write and empty on read.
Percussion lives on channel 10 (index 9).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import mido

from backline.score import TPQ, Note, Score, Tempo, TimeSignature, Track, TrackRole

PERCUSSION_CHANNEL = 9


def write_midi(score: Score, path: Path) -> None:
    """Write ``score`` as a type-1 MIDI file."""
    score_to_midi(score).save(path)


def score_to_midi(score: Score) -> mido.MidiFile:
    """Convert a score to a type-1 :class:`mido.MidiFile` at the score's resolution."""
    midi = mido.MidiFile(type=1, ticks_per_beat=score.tpq)
    starts = score.bar_starts()

    conductor: list[tuple[int, int, mido.Message | mido.MetaMessage]] = [
        (0, 0, mido.MetaMessage("track_name", name=score.title or "Backline"))
    ]
    for ts in score.time_signatures:
        conductor.append(
            (
                starts[ts.bar],
                1,
                mido.MetaMessage(
                    "time_signature", numerator=ts.numerator, denominator=ts.denominator
                ),
            )
        )
    for tempo in score.tempos:
        conductor.append(
            (tempo.tick, 2, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo.bpm)))
        )
    midi.tracks.append(_to_track(conductor, end=score.end))

    free_channels = iter(c for c in range(16) if c != PERCUSSION_CHANNEL)
    for track in score.tracks:
        channel = PERCUSSION_CHANNEL if track.is_percussion else next(free_channels)
        events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = [
            (0, 0, mido.MetaMessage("track_name", name=track.name))
        ]
        if not track.is_percussion:
            events.append(
                (0, 1, mido.Message("program_change", channel=channel, program=track.program))
            )
        for note in track.notes:
            events.append(
                (
                    note.start,
                    3,
                    mido.Message(
                        "note_on", channel=channel, note=note.pitch, velocity=note.velocity
                    ),
                )
            )
            events.append(
                (
                    note.end,
                    2,
                    mido.Message("note_off", channel=channel, note=note.pitch, velocity=0),
                )
            )
        midi.tracks.append(_to_track(events, end=score.end))
    return midi


def _to_track(
    events: list[tuple[int, int, mido.Message | mido.MetaMessage]], *, end: int
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    now = 0
    for tick, _, message in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(message.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=max(end - now, 0)))
    return track


@dataclass(slots=True)
class _OpenNote:
    start: int
    velocity: int


def read_midi(path: Path) -> Score:
    """Read a MIDI file into a score, rescaled to :data:`~backline.score.TPQ`."""
    return midi_to_score(mido.MidiFile(path), title=path.stem)


def midi_to_score(midi: mido.MidiFile, *, title: str = "") -> Score:
    """Convert a :class:`mido.MidiFile` to a score.

    Every (track, channel) pair with notes becomes a :class:`Track`. Roles are guessed
    from the channel and General MIDI program; the ingest stage refines them.
    """
    source_tpq = int(midi.ticks_per_beat)

    def scale(tick: int) -> int:
        return round(tick * TPQ / source_tpq)

    tempos: dict[int, float] = {}
    meters: dict[int, tuple[int, int]] = {}
    notes: dict[tuple[int, int], list[Note]] = defaultdict(list)
    programs: dict[tuple[int, int], int] = {}
    names: dict[int, str] = {}
    last_tick = 0

    for index, midi_track in enumerate(midi.tracks):
        now = 0
        open_notes: dict[tuple[int, int], deque[_OpenNote]] = defaultdict(deque)
        for message in midi_track:
            now += message.time
            tick = scale(now)
            last_tick = max(last_tick, tick)
            if message.type == "set_tempo":
                tempos[tick] = round(mido.tempo2bpm(message.tempo), 3)
            elif message.type == "time_signature":
                meters[tick] = (message.numerator, message.denominator)
            elif message.type == "track_name":
                names.setdefault(index, message.name)
            elif message.type == "program_change":
                programs.setdefault((index, message.channel), message.program)
            elif message.type == "note_on" and message.velocity > 0:
                open_notes[(message.channel, message.note)].append(
                    _OpenNote(tick, message.velocity)
                )
            elif message.type in ("note_off", "note_on"):
                queue = open_notes.get((message.channel, message.note))
                if queue:
                    opened = queue.popleft()
                    notes[(index, message.channel)].append(
                        Note(
                            start=opened.start,
                            pitch=message.note,
                            duration=max(tick - opened.start, 1),
                            velocity=opened.velocity,
                        )
                    )

    tempo_list = [Tempo(tick, bpm) for tick, bpm in sorted(tempos.items())]
    if not tempo_list or tempo_list[0].tick != 0:
        tempo_list.insert(0, Tempo(0, 120.0))

    end = max([last_tick, *(n.end for group in notes.values() for n in group)])
    time_signatures, n_bars = _meters_to_bars(meters, end)

    tracks = []
    for (index, channel), group in sorted(notes.items()):
        program = programs.get((index, channel), 0)
        percussion = channel == PERCUSSION_CHANNEL
        tracks.append(
            Track(
                name=names.get(index, f"Track {index}"),
                role=_guess_role(program, percussion),
                notes=sorted(group),
                program=program,
                is_percussion=percussion,
            )
        )
    return Score(
        n_bars=n_bars,
        time_signatures=time_signatures,
        tempos=tempo_list,
        tracks=tracks,
        title=names.get(0, title) if len(midi.tracks) > 1 else title,
    )


def _meters_to_bars(
    meters: dict[int, tuple[int, int]], end: int
) -> tuple[list[TimeSignature], int]:
    """Turn tick-positioned meter changes into bar-positioned ones, snapping to barlines."""
    changes = sorted(meters.items()) or [(0, (4, 4))]
    if changes[0][0] != 0:
        changes.insert(0, (0, (4, 4)))
    result: list[TimeSignature] = []
    bar, bar_start = 0, 0
    current = TimeSignature(0, *changes[0][1])
    for tick, (numerator, denominator) in changes[1:]:
        length = current.bar_ticks(TPQ)
        bars_elapsed = -(-(tick - bar_start) // length)  # round up to the next barline
        bar += bars_elapsed
        bar_start += bars_elapsed * length
        result.append(current)
        current = TimeSignature(bar, numerator, denominator)
    result.append(current)
    deduped: dict[int, TimeSignature] = {ts.bar: ts for ts in result}
    length = current.bar_ticks(TPQ)
    n_bars = bar + max(1, -(-(end - bar_start) // length))
    return sorted(deduped.values(), key=lambda ts: ts.bar), n_bars


def _guess_role(program: int, percussion: bool) -> TrackRole:
    if percussion:
        return TrackRole.DRUMS
    if 32 <= program <= 39:
        return TrackRole.BASS
    if 24 <= program <= 31:
        return TrackRole.GUITAR
    return TrackRole.OTHER
