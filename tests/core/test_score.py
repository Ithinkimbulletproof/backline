# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from backline.score import TPQ, Note, Score, Tempo, TimeSignature, Track, TrackRole


def _score(**kwargs: object) -> Score:
    defaults: dict[str, object] = {
        "n_bars": 4,
        "time_signatures": [TimeSignature(0, 4, 4), TimeSignature(2, 3, 4)],
        "tempos": [Tempo(0, 120.0), Tempo(3840, 140.0)],
    }
    defaults.update(kwargs)
    return Score(**defaults)  # type: ignore[arg-type]


def test_note_validation() -> None:
    with pytest.raises(ValueError, match="duration"):
        Note(start=0, pitch=40, duration=0)
    with pytest.raises(ValueError, match="pitch"):
        Note(start=0, pitch=128, duration=10)
    with pytest.raises(ValueError, match="velocity"):
        Note(start=0, pitch=40, duration=10, velocity=0)
    assert Note(start=10, pitch=40, duration=5).end == 15
    assert Note(start=0, pitch=40, duration=5).transposed(-12).pitch == 28


def test_time_signature_lengths() -> None:
    assert TimeSignature(0, 4, 4).bar_ticks() == 4 * TPQ
    assert TimeSignature(0, 6, 8).bar_ticks() == 3 * TPQ
    assert TimeSignature(0, 3, 4).beat_ticks() == TPQ
    with pytest.raises(ValueError, match="power of two"):
        TimeSignature(0, 4, 3)


def test_bar_grid_follows_meter_changes() -> None:
    score = _score()
    assert score.bar_starts() == [0, 3840, 7680, 10560, 13440]
    assert score.end == 13440
    assert score.bar_of(0) == 0
    assert score.bar_of(3840) == 1
    assert score.bar_of(13439) == 3
    with pytest.raises(ValueError, match="outside"):
        score.bar_of(13440)
    assert str(score.time_signature_at(3)) == "3/4"


def test_tempo_map() -> None:
    score = _score()
    assert score.tempo_at(0) == 120.0
    assert score.tempo_at(3839) == 120.0
    assert score.tempo_at(5000) == 140.0


def test_score_validation() -> None:
    with pytest.raises(ValueError, match="bar 0"):
        _score(time_signatures=[TimeSignature(1, 4, 4)])
    with pytest.raises(ValueError, match="tick 0"):
        _score(tempos=[Tempo(10, 120.0)])
    with pytest.raises(ValueError, match="sorted"):
        _score(tempos=[Tempo(0, 120.0), Tempo(5, 100.0), Tempo(5, 90.0)])
    with pytest.raises(ValueError, match="at least one bar"):
        _score(n_bars=0)


def test_tracks_with_role() -> None:
    score = _score(
        tracks=[
            Track("Gtr", TrackRole.GUITAR),
            Track("Bass", TrackRole.BASS),
            Track("Gtr 2", TrackRole.GUITAR),
        ]
    )
    assert [t.name for t in score.tracks_with_role(TrackRole.GUITAR)] == ["Gtr", "Gtr 2"]
    assert score.tracks_with_role(TrackRole.DRUMS) == []
