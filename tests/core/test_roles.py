# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from backline.config import BacklineConfig
from backline.ingest.roles import classify_track, rank_guitars
from backline.score import TPQ, Note, Score, Tempo, TimeSignature, Track, TrackRole

E = TPQ // 2


def _notes(pitches: list[int], step: int = E) -> list[Note]:
    return [Note(i * step, p, step) for i, p in enumerate(pitches)]


def test_clear_cases(tiny_config: BacklineConfig) -> None:
    hints = tiny_config.ingest.name_hints
    bass = Track("Bass", TrackRole.OTHER, _notes([28, 31, 33]), 33, tuning=(28, 33, 38, 43))
    gp7_bass = Track("Track 2", TrackRole.OTHER, _notes([28]), 0, instrument="electricBass")
    guitar = Track(
        "Rhythm Gtr", TrackRole.OTHER, _notes([40, 47, 52]), 30, tuning=(40, 45, 50, 55, 59, 64)
    )
    drums = Track("Kit", TrackRole.OTHER, _notes([36, 38]), is_percussion=True)
    keys = Track("Piano", TrackRole.OTHER, _notes([60, 64]), 0, instrument="acousticPiano")
    for track, role in (
        (bass, TrackRole.BASS),
        (gp7_bass, TrackRole.BASS),
        (guitar, TrackRole.GUITAR),
        (drums, TrackRole.DRUMS),
        (keys, TrackRole.OTHER),
    ):
        decision = classify_track(track, hints)
        assert decision.role is role, (track.name, decision)
        assert not decision.ambiguous, (track.name, decision)


def test_conflicting_clues_are_flagged(tiny_config: BacklineConfig) -> None:
    hints = tiny_config.ingest.name_hints
    mislabelled = Track("Track 3", TrackRole.OTHER, _notes([28, 33]), 30, tuning=(28, 33, 38, 43))
    decision = classify_track(mislabelled, hints)
    assert decision.role is TrackRole.BASS
    assert decision.ambiguous
    assert any("tuning" in reason for reason in decision.reasons)


def test_track_without_clues(tiny_config: BacklineConfig) -> None:
    decision = classify_track(Track("x", TrackRole.OTHER), tiny_config.ingest.name_hints)
    assert decision.role is TrackRole.OTHER
    assert decision.ambiguous


def test_rhythm_guitar_beats_lead(tiny_config: BacklineConfig) -> None:
    hints = tiny_config.ingest.name_hints
    power_chords = [Note(i * E, p, E) for i in range(16) for p in (40, 47, 52)]
    rhythm = Track("Guitar 2", TrackRole.GUITAR, power_chords, 30)
    lead = Track("Guitar 1", TrackRole.GUITAR, _notes([76, 79, 81, 79] * 4), 30)
    score = Score(2, [TimeSignature(0, 4, 4)], [Tempo(0, 120.0)], [lead, rhythm])
    ranked = rank_guitars(score, [0, 1], hints)
    assert [p.index for p in ranked] == [1, 0]
    assert ranked[1].lead_like
    assert not ranked[0].lead_like
    assert ranked[0].chord_share == 1.0

    named = Score(
        2,
        [TimeSignature(0, 4, 4)],
        [Tempo(0, 120.0)],
        [
            Track("Rhythm", TrackRole.GUITAR, _notes([76] * 8), 30),
            Track("Solo", TrackRole.GUITAR, _notes([76] * 8), 30),
        ],
    )
    assert [p.index for p in rank_guitars(named, [0, 1], hints)] == [0, 1]


def test_profile_over_a_bar_range(tiny_config: BacklineConfig) -> None:
    hints = tiny_config.ingest.name_hints
    notes = [Note(i * E, p, E) for i in range(8) for p in (40, 47)]
    score = Score(
        4, [TimeSignature(0, 4, 4)], [Tempo(0, 120.0)], [Track("Gtr", TrackRole.GUITAR, notes, 30)]
    )
    (first,) = rank_guitars(score, [0], hints, bars=range(0, 1))
    (late,) = rank_guitars(score, [0], hints, bars=range(2, 4))
    assert first.coverage == 1.0
    assert late.notes == 0
