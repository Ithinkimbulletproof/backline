# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from backline.score import TPQ, Note, Score, Tempo, TimeSignature, Track, TrackRole
from backline.timing import Feel, apply_feel, feel_position, swing_tick

E = TPQ // 2
S = TPQ // 4


@pytest.mark.parametrize(
    ("feel", "pos", "expected"),
    [
        (Feel.STRAIGHT, E, E),
        (Feel.TRIPLET_8, 0, 0),
        (Feel.TRIPLET_8, E, 2 * TPQ // 3),
        (Feel.TRIPLET_8, S, TPQ // 3),
        (Feel.TRIPLET_8, 3 * S, 2 * TPQ // 3 + TPQ // 6),
        (Feel.TRIPLET_8, TPQ, TPQ),
        (Feel.TRIPLET_8, TPQ + E, TPQ + 2 * TPQ // 3),
        (Feel.DOTTED_8, E, 3 * TPQ // 4),
        (Feel.SCOTTISH_8, E, TPQ // 4),
        (Feel.TRIPLET_16, S, 2 * E // 3),
        (Feel.TRIPLET_16, E, E),
        (Feel.DOTTED_16, S, 3 * E // 4),
    ],
)
def test_feel_positions(feel: Feel, pos: int, expected: int) -> None:
    assert feel_position(pos, feel) == expected


def test_incomplete_pair_at_bar_end_is_left_alone() -> None:
    bar_ticks = 7 * E  # 7/8
    assert feel_position(6 * E, Feel.TRIPLET_8, bar_ticks) == 6 * E
    assert feel_position(5 * E, Feel.TRIPLET_8, bar_ticks) == 4 * E + 2 * TPQ // 3


def test_swing_tick_matches_triplet_feel() -> None:
    assert swing_tick(3 * TPQ + E) == 3 * TPQ + 2 * TPQ // 3


def test_apply_feel_moves_only_marked_bars() -> None:
    eighths = [Note(i * E, 40, E) for i in range(16)]
    score = Score(
        n_bars=2,
        time_signatures=[TimeSignature(0, 4, 4)],
        tempos=[Tempo(0, 120.0)],
        tracks=[Track("Gtr", TrackRole.GUITAR, list(eighths))],
    )
    apply_feel(score, {0: Feel.TRIPLET_8})
    notes = score.tracks[0].notes
    first_bar, second_bar = notes[:8], notes[8:]
    assert [n.start for n in first_bar[:3]] == [0, 640, 960]
    assert [n.duration for n in first_bar[:2]] == [640, 320]
    assert [n.start for n in second_bar] == [4 * TPQ + i * E for i in range(8)]
    assert all(n.duration == E for n in second_bar)


def test_apply_feel_without_feels_is_a_no_op() -> None:
    notes = [Note(E, 40, E)]
    score = Score(
        n_bars=1,
        time_signatures=[TimeSignature(0, 4, 4)],
        tempos=[Tempo(0, 120.0)],
        tracks=[Track("Gtr", TrackRole.GUITAR, list(notes))],
    )
    apply_feel(score, {0: Feel.STRAIGHT})
    assert score.tracks[0].notes == notes
