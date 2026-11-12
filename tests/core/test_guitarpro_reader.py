# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

import guitarpro as gp
import pytest

from backline.config import BacklineConfig
from backline.formats.common import ReadStats, ScoreReadError
from backline.formats.guitarpro import read_guitarpro, score_to_gp5, write_gp5
from backline.score import TPQ, Note, NoteFlag, Score, Tempo, TimeSignature, Track, TrackRole
from backline.synth import SynthSong
from tests.support import note_signature

E = TPQ // 2


def _straight_riff(n_bars: int = 2) -> Score:
    return Score(
        n_bars=n_bars,
        time_signatures=[TimeSignature(0, 4, 4)],
        tempos=[Tempo(0, 110.0)],
        tracks=[
            Track(
                "Rhythm",
                TrackRole.GUITAR,
                [Note(i * E, 40, E, 95) for i in range(8 * n_bars)],
                program=30,
                tuning=(40, 45, 50, 55, 59, 64),
            )
        ],
        title="Riff",
        artist="Nobody",
    )


def test_round_trip_of_synthetic_songs(
    corpus: list[SynthSong], tiny_config: BacklineConfig, tmp_path: Path
) -> None:
    for index, song in enumerate(corpus[:10]):
        path = tmp_path / f"{index}.gp5"
        write_gp5(song.score, path)
        stats = ReadStats()
        back = read_guitarpro(path, stats=stats)
        assert note_signature(back, tiny_config.velocity) == note_signature(
            song.score, tiny_config.velocity
        )
        assert back.title == song.score.title
        assert back.artist == song.score.artist
        assert back.n_bars == song.score.n_bars
        assert back.time_signatures == song.score.time_signatures
        assert back.tempos == [Tempo(0, round(song.score.tempos[0].bpm))]
        assert [t.role for t in back.tracks] == [t.role for t in song.score.tracks]
        assert [t.tuning for t in back.tracks] == [t.tuning for t in song.score.tracks]
        assert [t.is_percussion for t in back.tracks] == [
            t.is_percussion for t in song.score.tracks
        ]
        assert stats == ReadStats()


def test_triplet_feel_is_applied(tmp_path: Path) -> None:
    song = score_to_gp5(_straight_riff())
    song.measureHeaders[0].tripletFeel = gp.TripletFeel.eighth
    path = tmp_path / "shuffle.gp5"
    gp.write(song, str(path))

    swung = read_guitarpro(path).tracks[0].notes
    assert [n.start for n in swung[:4]] == [0, 640, 960, 1600]
    assert [n.start for n in swung[8:10]] == [4 * TPQ, 4 * TPQ + E]

    straight = read_guitarpro(path, apply_triplet_feel=False).tracks[0].notes
    assert [n.start for n in straight[:2]] == [0, E]


def test_tempo_changes_and_capo(tmp_path: Path) -> None:
    song = score_to_gp5(_straight_riff())
    beat = song.tracks[0].measures[1].voices[0].beats[0]
    beat.effect.mixTableChange = gp.MixTableChange(tempo=gp.MixTableItem(value=90, duration=0))
    song.tracks[0].offset = 2
    path = tmp_path / "tempo.gp5"
    gp.write(song, str(path))

    score = read_guitarpro(path)
    assert score.tempos == [Tempo(0, 110.0), Tempo(4 * TPQ, 90.0)]
    assert {n.pitch for n in score.tracks[0].notes} == {42}


def test_grace_notes_are_dropped_and_counted(tmp_path: Path) -> None:
    song = score_to_gp5(_straight_riff(1))
    first = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
    first.effect.grace = gp.GraceEffect(fret=3)
    path = tmp_path / "grace.gp5"
    gp.write(song, str(path))

    stats = ReadStats()
    score = read_guitarpro(path, stats=stats)
    assert stats["grace_notes_dropped"] == 1
    assert len(score.tracks[0].notes) == 8


def test_flags_survive(tmp_path: Path) -> None:
    score = _straight_riff(1)
    score.tracks[0].notes = [
        Note(0, 40, E, 95, NoteFlag.PALM_MUTE),
        Note(E, 40, E, 95, NoteFlag.DEAD),
        Note(2 * E, 40, E, 31, NoteFlag.GHOST),
        Note(3 * E, 40, E, 111, NoteFlag.ACCENT),
        Note(4 * E, 40, 4 * E, 127, NoteFlag.HEAVY_ACCENT),
    ]
    path = tmp_path / "flags.gp5"
    write_gp5(score, path)
    back = read_guitarpro(path).tracks[0].notes
    assert [n.flags for n in back] == [n.flags for n in score.tracks[0].notes]
    assert back[-1].duration == 4 * E


@pytest.mark.parametrize(("suffix", "version"), [(".gp4", (4, 0, 6)), (".gp3", (3, 0, 0))])
def test_older_formats(tmp_path: Path, suffix: str, version: tuple[int, int, int]) -> None:
    path = tmp_path / f"old{suffix}"
    gp.write(score_to_gp5(_straight_riff(1)), str(path), version=version)
    score = read_guitarpro(path)
    assert [n.start for n in score.tracks[0].notes] == [i * E for i in range(8)]


def test_corrupt_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.gp5"
    path.write_bytes(b"FICHIER GUITAR PRO v5.10" + b"\x00" * 7)
    with pytest.raises(ScoreReadError, match=r"broken\.gp5"):
        read_guitarpro(path)
