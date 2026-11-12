# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from collections import Counter
from pathlib import Path

import guitarpro as gp
import mido
import pytest

from backline.formats.guitarpro import write_gp5
from backline.formats.midi import read_midi, write_midi
from backline.formats.tablature import GRID, TabLayoutError, split_ticks
from backline.score import TPQ, Note, NoteFlag, Score, Tempo, TimeSignature, Track, TrackRole
from backline.synth import SynthSong


def _gp_notes(track: gp.Track) -> list[gp.Note]:
    return [
        note
        for measure in track.measures
        for beat in measure.voices[0].beats
        for note in beat.notes
        if note.type != gp.NoteType.tie
    ]


# --------------------------------------------------------------------------- MIDI


def test_midi_round_trip(corpus: list[SynthSong], tmp_path: Path) -> None:
    for index, song in enumerate(corpus[:8]):
        path = tmp_path / f"{index}.mid"
        write_midi(song.score, path)
        back = read_midi(path)
        assert back.n_bars == song.score.n_bars
        assert back.time_signatures == song.score.time_signatures
        assert back.tempos[0].bpm == pytest.approx(song.score.tempos[0].bpm)
        assert [t.role for t in back.tracks] == [t.role for t in song.score.tracks]
        for original, restored in zip(song.score.tracks, back.tracks, strict=True):
            key = lambda n: (n.start, n.pitch, n.duration, n.velocity)  # noqa: E731
            assert sorted(map(key, restored.notes)) == sorted(map(key, original.notes))
            assert restored.is_percussion == original.is_percussion


def test_midi_is_rescaled_to_internal_resolution(tmp_path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=96)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=30, time=0))
    track.append(mido.Message("note_on", note=40, velocity=100, time=96))
    track.append(mido.Message("note_off", note=40, velocity=0, time=48))
    midi.tracks.append(track)
    path = tmp_path / "riff.mid"
    midi.save(path)

    score = read_midi(path)
    (gtr,) = score.tracks
    assert gtr.role is TrackRole.GUITAR
    assert gtr.notes == [Note(start=TPQ, pitch=40, duration=TPQ // 2, velocity=100)]
    assert score.tempos == [Tempo(0, 120.0)]


def test_midi_meter_changes_land_on_barlines(tmp_path: Path) -> None:
    score = Score(
        n_bars=4,
        time_signatures=[TimeSignature(0, 4, 4), TimeSignature(2, 3, 4)],
        tempos=[Tempo(0, 100.0)],
        tracks=[Track("Bass", TrackRole.BASS, [Note(0, 28, 480)], program=33)],
    )
    path = tmp_path / "meters.mid"
    write_midi(score, path)
    back = read_midi(path)
    assert back.time_signatures == score.time_signatures
    assert back.n_bars == 4


# ---------------------------------------------------------------------------- GP5


def test_gp5_round_trip_through_pyguitarpro(corpus: list[SynthSong], tmp_path: Path) -> None:
    for index, song in enumerate(corpus[:8]):
        path = tmp_path / f"{index}.gp5"
        write_gp5(song.score, path)
        parsed = gp.parse(str(path))
        assert parsed.title == song.score.title
        assert parsed.artist == song.score.artist
        assert parsed.tempo == round(song.score.tempos[0].bpm)
        assert len(parsed.measureHeaders) == song.score.n_bars
        for original, gp_track in zip(song.score.tracks, parsed.tracks, strict=True):
            notes = _gp_notes(gp_track)
            assert Counter(n.realValue for n in notes) == Counter(n.pitch for n in original.notes)
            assert gp_track.isPercussionTrack == original.is_percussion
            ghosts = sum(bool(n.flags & NoteFlag.GHOST) for n in original.notes)
            assert sum(n.effect.ghostNote for n in notes) == ghosts
            muted = sum(bool(n.flags & NoteFlag.PALM_MUTE) for n in original.notes)
            assert sum(n.effect.palmMute for n in notes) == muted
            if original.tuning:
                assert [s.value for s in gp_track.strings] == list(reversed(original.tuning))


def test_gp5_ties_notes_across_barlines(tmp_path: Path) -> None:
    score = Score(
        n_bars=2,
        time_signatures=[TimeSignature(0, 4, 4)],
        tempos=[Tempo(0, 120.0)],
        tracks=[
            Track("Bass", TrackRole.BASS, [Note(3 * TPQ, 33, 2 * TPQ)], tuning=(28, 33, 38, 43))
        ],
    )
    path = tmp_path / "tie.gp5"
    write_gp5(score, path)
    track = gp.parse(str(path)).tracks[0]
    first, second = (m.voices[0].beats for m in track.measures)
    assert [b.status for b in first] == [gp.BeatStatus.rest, gp.BeatStatus.normal]
    assert first[1].notes[0].type == gp.NoteType.normal
    assert second[0].notes[0].type == gp.NoteType.tie
    assert second[0].notes[0].realValue == 33


@pytest.mark.parametrize("ticks", [GRID, 240, 320, 400, 1200, 1440, 2880, 3840, 5 * GRID * 7])
def test_split_ticks_adds_up(ticks: int) -> None:
    pieces = split_ticks(ticks)
    assert sum(p.ticks for p in pieces) == ticks
    assert pieces == tuple(sorted(pieces, key=lambda p: -p.ticks))


def test_split_ticks_prefers_plain_durations() -> None:
    assert [p.ticks for p in split_ticks(3840)] == [3840]
    assert [p.ticks for p in split_ticks(1200)] == [960, 240]
    assert [(p.ticks, p.dotted) for p in split_ticks(1440)] == [(1440, True)]


def test_split_ticks_rejects_off_grid_lengths() -> None:
    with pytest.raises(TabLayoutError):
        split_ticks(30)
