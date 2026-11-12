# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Token encoding, decoding, segmentation and transposition."""

from __future__ import annotations

import bisect

import numpy as np
import pytest

from backline.config import BacklineConfig
from backline.music import Dynamic
from backline.score import TrackRole
from backline.synth import SynthSong, generate_corpus
from backline.theory.annotate import Annotation, annotate_score
from backline.tokenizer.augment import Transposer
from backline.tokenizer.encode import EncodeError, SegmentEncoder, SegmentSource, decode
from backline.tokenizer.segments import song_segments
from backline.tokenizer.vocab import STEP, Vocab, snap_duration, tempo_bucket


@pytest.fixture(scope="module")
def annotated(tiny_config: BacklineConfig) -> list[tuple[SynthSong, Annotation]]:
    songs = generate_corpus(tiny_config, n_songs=16, seed=31)
    return [(s, annotate_score(s.score)) for s in songs]


@pytest.fixture(scope="module")
def vocab(tiny_config: BacklineConfig) -> Vocab:
    return Vocab.from_config(tiny_config)


def test_vocabulary_is_deterministic(tiny_config: BacklineConfig, vocab: Vocab) -> None:
    again = Vocab.from_config(tiny_config)
    assert again.tokens == vocab.tokens
    assert Vocab.from_json(vocab.to_json()).fingerprint() == vocab.fingerprint()
    assert len(vocab) < 600
    for token in ("DRUM_crash2", "ROLE_approach", "KEY_4_minor"):
        assert token in vocab


def test_helpers() -> None:
    assert snap_duration(0) == 1
    assert snap_duration(25) == 24
    assert snap_duration(100) == 96
    edges = (60, 120, 180)
    assert [tempo_bucket(b, edges) for b in (50, 60, 61, 200)] == [0, 0, 1, 3]


def _segment_notes(
    song: SynthSong, track_index: int, source: SegmentSource
) -> list[tuple[int, int, int]]:
    score = song.score
    starts = score.bar_starts()
    lo, hi = starts[source.start_bar], starts[source.start_bar + source.n_bars]
    return sorted(
        (round((n.start - lo) / STEP), n.pitch, n.duration)
        for n in score.tracks[track_index].notes
        if lo <= n.start < hi
    )


def test_round_trip_of_every_stream(
    annotated: list[tuple[SynthSong, Annotation]], tiny_config: BacklineConfig, vocab: Vocab
) -> None:
    encoder = SegmentEncoder(tiny_config, vocab)
    checked = 0
    for song, annotation in annotated:
        for source in list(song_segments(song.score, annotation, tiny_config))[:3]:
            encoded = encoder.encode(source)
            decoded = decode(encoded.tokens, vocab)
            bar_steps = song.score.time_signature_at(source.start_bar).bar_ticks() // STEP
            key = annotation.key_at_bar(source.start_bar)
            assert (decoded.tonic, decoded.mode) == (key.tonic, key.mode)

            bass = sorted((n.bar * bar_steps + n.pos, n.pitch) for n in decoded.bass)
            truth = [(s, p) for s, p, _ in _segment_notes(song, source.bass, source)]
            assert bass == truth
            guitar = sorted((n.bar * bar_steps + n.pos, n.pitch) for n in decoded.guitar)
            assert guitar == [(s, p) for s, p, _ in _segment_notes(song, source.guitar, source)]
            assert all(n.role is not None for n in decoded.bass)

            kit = song.score.tracks[source.drums]
            starts = song.score.bar_starts()
            lo, hi = starts[source.start_bar], starts[source.start_bar + source.n_bars]
            expected = {
                (round((n.start - lo) / STEP), tiny_config.drums.class_of(n.pitch))
                for n in kit.notes
                if lo <= n.start < hi
            }
            assert {(h.bar * bar_steps + h.pos, h.drum_class) for h in decoded.drums} == expected

            chord_starts = [c.start for c in annotation.chords]
            for bar in range(source.n_bars):
                tick = starts[source.start_bar + bar]
                chord = annotation.chords[bisect.bisect_right(chord_starts, tick) - 1]
                first = next(c for c in decoded.plan if c.bar == bar and c.pos == 0)
                assert (first.root, first.quality) == (chord.root, chord.quality)
            checked += 1
    assert checked > 10


def test_levels_survive(
    annotated: list[tuple[SynthSong, Annotation]], tiny_config: BacklineConfig, vocab: Vocab
) -> None:
    encoder = SegmentEncoder(tiny_config, vocab)
    levels = set()
    for song, annotation in annotated:
        for source in song_segments(song.score, annotation, tiny_config):
            levels |= {h.level for h in decode(encoder.encode(source).tokens, vocab).drums}
    assert {Dynamic.GHOST, Dynamic.NORMAL, Dynamic.ACCENT} <= levels


def test_condition_is_the_prefix(
    annotated: list[tuple[SynthSong, Annotation]], tiny_config: BacklineConfig, vocab: Vocab
) -> None:
    encoder = SegmentEncoder(tiny_config, vocab)
    song, annotation = annotated[0]
    source = next(song_segments(song.score, annotation, tiny_config))
    encoded = encoder.encode(source)
    bars = range(source.start_bar, source.start_bar + source.n_bars)
    prompt = encoder.condition(song.score, source.guitar, bars, annotation.key_at_bar(bars.start))
    assert encoded.tokens[: len(prompt)] == prompt
    assert vocab.token(prompt[-1]) == "SEP"
    assert vocab.token(encoded.tokens[encoded.target_start]) == "PLAN"


def test_lead_guitars_never_carry_the_riff(tiny_config: BacklineConfig) -> None:
    songs = [
        s for s in generate_corpus(tiny_config, n_songs=60, seed=8) if len(s.score.tracks) == 4
    ]
    assert songs
    for song in songs:
        annotation = annotate_score(song.score)
        lead = next(
            i for i, t in enumerate(song.score.tracks) if t.role is TrackRole.GUITAR and i > 0
        )
        assert all(s.guitar != lead for s in song_segments(song.score, annotation, tiny_config))


def test_segments_cover_configured_lengths(
    annotated: list[tuple[SynthSong, Annotation]], tiny_config: BacklineConfig
) -> None:
    lengths = {s.n_bars for song, a in annotated for s in song_segments(song.score, a, tiny_config)}
    assert lengths == set(tiny_config.tokenizer.segment_bars)


def test_meter_change_is_rejected(
    annotated: list[tuple[SynthSong, Annotation]], tiny_config: BacklineConfig, vocab: Vocab
) -> None:
    from backline.score import TimeSignature

    song, annotation = annotated[0]
    score = song.score
    changed = type(score)(
        score.n_bars,
        [TimeSignature(0, 4, 4), TimeSignature(1, 3, 4)],
        score.tempos,
        score.tracks,
    )
    source = SegmentSource(changed, annotation, 0, 1, 2, 0, 2)
    with pytest.raises(EncodeError, match="meter"):
        SegmentEncoder(tiny_config, vocab).encode(source)


def test_transposition(
    annotated: list[tuple[SynthSong, Annotation]], tiny_config: BacklineConfig, vocab: Vocab
) -> None:
    encoder = SegmentEncoder(tiny_config, vocab)
    transposer = Transposer(vocab, tiny_config)
    song, annotation = annotated[1]
    source = next(song_segments(song.score, annotation, tiny_config))
    encoded = encoder.encode(source)
    allowed = transposer.allowed(encoded.guitar_range, encoded.bass_range)
    assert 0 in allowed
    shift = max(allowed)
    moved = decode(transposer.apply(np.array(encoded.tokens), shift).tolist(), vocab)
    original = decode(encoded.tokens, vocab)
    assert [n.pitch + shift for n in original.bass] == [n.pitch for n in moved.bass]
    assert [n.pitch + shift for n in original.guitar] == [n.pitch for n in moved.guitar]
    assert moved.tonic == (original.tonic + shift) % 12
    assert [c.degree for c in moved.plan] == [c.degree for c in original.plan]
    assert [c.root for c in moved.plan] == [
        None if c.root is None else (c.root + shift) % 12 for c in original.plan
    ]
    assert moved.drums == original.drums


def test_allowed_shifts_respect_ranges(tiny_config: BacklineConfig, vocab: Vocab) -> None:
    transposer = Transposer(vocab, tiny_config)
    low, high = tiny_config.instruments.bass.train_range
    assert min(transposer.allowed((40, 60), (low, low + 12))) == 0
    assert max(transposer.allowed((40, 60), (high - 12, high))) == 0
