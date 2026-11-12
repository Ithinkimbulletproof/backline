# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Harmonic analysis on hand-built examples and on the synthetic corpus."""

from __future__ import annotations

import bisect

import pytest

from backline.config import BacklineConfig
from backline.music import BassRole, ChordQuality, HarmonicFunction, Mode, parse_pitch
from backline.score import TPQ, Note, Score, Tempo, TimeSignature, Track, TrackRole
from backline.synth import generate_corpus
from backline.theory.annotate import (
    Annotation,
    annotate_score,
    annotation_from_dict,
    annotation_to_dict,
)
from backline.theory.evaluate import evaluate
from backline.theory.grid import beat_grid, beats_per_bar

Q, E = TPQ, TPQ // 2
BAR = 4 * Q
SHAPES = {
    ChordQuality.POWER: (0, 7, 12),
    ChordQuality.MAJOR: (0, 7, 12, 16),
    ChordQuality.MINOR: (0, 7, 12, 15),
    ChordQuality.DOMINANT7: (0, 7, 10, 16),
}


def _song(
    chords: list[tuple[str, ChordQuality, int]],
    bass_lines: list[list[str]] | None = None,
    meter: tuple[int, int] = (4, 4),
) -> Score:
    """Guitar strums each chord in 8ths; the bass plays roots in 8ths unless a line is given.

    ``chords`` are (root name in octave 2, quality, length in bars).
    """
    guitar: list[Note] = []
    bass: list[Note] = []
    bar_ticks = meter[0] * TPQ * 4 // meter[1]
    tick = 0
    bar = 0
    for index, (name, quality, bars) in enumerate(chords):
        root = parse_pitch(name)
        for offset in range(0, bars * bar_ticks, E):
            guitar.extend(Note(tick + offset, root + i, E) for i in SHAPES[quality])
        line = bass_lines[index] if bass_lines else None
        if line:
            step = bars * bar_ticks // len(line)
            bass.extend(Note(tick + k * step, parse_pitch(p), step) for k, p in enumerate(line))
        else:
            bass.extend(Note(tick + o, root - 12, E) for o in range(0, bars * bar_ticks, E))
        tick += bars * bar_ticks
        bar += bars
    return Score(
        n_bars=bar,
        time_signatures=[TimeSignature(0, *meter)],
        tempos=[Tempo(0, 120.0)],
        tracks=[
            Track("Guitar", TrackRole.GUITAR, guitar, 30),
            Track("Bass", TrackRole.BASS, bass, 33),
        ],
    )


def _chord_at(annotation: Annotation, tick: int) -> tuple[int | None, ChordQuality | None]:
    starts = [c.start for c in annotation.chords]
    chord = annotation.chords[bisect.bisect_right(starts, tick) - 1]
    return chord.root, chord.quality


P = ChordQuality.POWER


def test_power_chords_stay_root_plus_five_and_leave_the_mode_open() -> None:
    score = _song([("E2", P, 1), ("G2", P, 1), ("A2", P, 1), ("E2", P, 1)] * 2)
    annotation = annotate_score(score)
    assert [_chord_at(annotation, b * BAR) for b in range(4)] == [(4, P), (7, P), (9, P), (4, P)]
    (key,) = annotation.keys
    assert key.tonic == 4
    assert not key.mode_certain


def test_blues_notes_do_not_break_the_key() -> None:
    blues = [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7]
    chords = [(["A2", "D3", "E2"][[0, 5, 7].index(d)], ChordQuality.DOMINANT7, 1) for d in blues]
    lines = []
    for degree in blues:
        root = parse_pitch("A1") + degree
        # boogie with a blue minor third: R b3 3 5 6 b7 6 5
        lines.append(
            [
                f"{p}"
                for p in (
                    root,
                    root + 3,
                    root + 4,
                    root + 7,
                    root + 9,
                    root + 10,
                    root + 9,
                    root + 7,
                )
            ]
        )
    from backline.music import pitch_name

    lines = [[pitch_name(int(p)) for p in line] for line in lines]
    score = _song(chords, lines)
    annotation = annotate_score(score)
    assert {k.tonic for k in annotation.keys} == {9}
    assert all(k.mode in (Mode.MAJOR, Mode.MIXOLYDIAN) for k in annotation.keys)
    roots = [_chord_at(annotation, b * BAR)[0] for b in range(12)]
    assert roots == [(9 + d) % 12 for d in blues]
    bass = score.tracks[1].sorted_notes()
    blue = [
        r
        for n, r in zip(bass, annotation.bass_roles, strict=True)
        if (n.pitch - 33) % 12 == 3 and (n.start // BAR) in (0, 1, 2, 3)
    ]
    assert blue
    assert set(blue) == {BassRole.BLUE_NOTE}


def test_double_plagal_cadence_reads_as_bvii_iv_i() -> None:
    maj = ChordQuality.MAJOR
    score = _song([("D2", maj, 1), ("A2", maj, 1), ("E2", maj, 2)] * 3)
    annotation = annotate_score(score)
    assert {k.tonic for k in annotation.keys} == {4}
    labelled = annotation.labelled_chords()
    degrees = [c.degree for c in labelled[:3]]
    functions = [c.function for c in labelled[:3]]
    assert degrees == [10, 5, 0]
    assert functions == [
        HarmonicFunction.MODAL,
        HarmonicFunction.SUBDOMINANT,
        HarmonicFunction.TONIC,
    ]


def test_chromatic_approach_to_the_tonic() -> None:
    lines = [
        ["E1", "E1", "E1", "E1", "E1", "E1", "E1", "E1"],
        ["A1", "A1", "A1", "A1", "A1", "A1", "A1", "A1"],
        ["B1", "B1", "B1", "B1", "B1", "B1", "B1", "D#2"],
        ["E2", "E2", "E2", "E2", "E2", "E2", "E2", "E2"],
    ]
    score = _song([("E2", P, 1), ("A2", P, 1), ("B2", P, 1), ("E2", P, 1)], lines)
    annotation = annotate_score(score)
    bass = score.tracks[1].sorted_notes()
    roles = dict(zip([(n.start, n.pitch) for n in bass], annotation.bass_roles, strict=True))
    assert roles[(2 * BAR + 7 * E, parse_pitch("D#2"))] is BassRole.CHROMATIC_APPROACH
    assert roles[(0, parse_pitch("E1"))] is BassRole.CHORD_TONE


def test_passing_and_neighbour_notes() -> None:
    lines = [["E1", "F#1", "G#1", "A1"], ["A1", "B1", "A1", "A1"]]
    score = _song([("E2", ChordQuality.MAJOR, 1), ("A2", ChordQuality.MAJOR, 1)], lines)
    annotation = annotate_score(score)
    roles = annotation.bass_roles
    assert roles[1] is BassRole.PASSING
    assert roles[5] is BassRole.NEIGHBOR


def test_modulation_up_a_whole_step_is_found() -> None:
    maj, m = ChordQuality.MAJOR, ChordQuality.MINOR
    first = [("E2", m, 1), ("C3", maj, 1), ("D3", maj, 1), ("E2", m, 1)] * 3
    second = [("F#2", m, 1), ("D3", maj, 1), ("E2", maj, 1), ("F#2", m, 1)] * 3
    annotation = annotate_score(_song(first + second))
    assert [(k.start_bar, k.tonic, k.mode) for k in annotation.keys] == [
        (0, 4, Mode.MINOR),
        (12, 6, Mode.MINOR),
    ]
    assert all(k.mode_certain for k in annotation.keys)


def test_beat_grid_handles_compound_meters() -> None:
    assert beats_per_bar(TimeSignature(0, 6, 8)) == 2
    assert beats_per_bar(TimeSignature(0, 7, 8)) == 7
    score = Score(2, [TimeSignature(0, 12, 8)], [Tempo(0, 90.0)])
    beats = beat_grid(score)
    assert len(beats) == 8
    assert beats[1].start == 3 * E


def test_annotation_round_trip() -> None:
    annotation = annotate_score(_song([("E2", P, 2), ("A2", ChordQuality.MAJOR, 2)]))
    assert annotation_from_dict(annotation_to_dict(annotation)) == annotation


def test_riff_without_bass_still_gets_a_key() -> None:
    score = _song([("E2", P, 1), ("D2", P, 1), ("A2", P, 1), ("E2", P, 1)] * 2)
    score.tracks = score.tracks[:1]
    annotation = annotate_score(score)
    assert annotation.bass_roles == []
    assert annotation.keys[0].tonic == 4


@pytest.mark.slow
def test_accuracy_on_the_synthetic_corpus(tiny_config: BacklineConfig) -> None:
    scores = evaluate(generate_corpus(tiny_config, n_songs=80, seed=23))
    assert scores.root >= 0.93
    assert scores.third >= 0.9
    assert scores.tonic >= 0.85
    assert scores.mode >= 0.7
    assert scores.bass_roles >= 0.9
