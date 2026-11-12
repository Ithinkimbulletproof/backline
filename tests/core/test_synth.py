# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import itertools
from collections import Counter

from backline.config import BacklineConfig
from backline.music import BassRole, ChordQuality
from backline.score import Track, TrackRole
from backline.synth import SynthSong, generate_corpus
from backline.synth.parts import swing_tick


def _track(song: SynthSong, role: TrackRole) -> Track:
    (track,) = song.score.tracks_with_role(role)
    return track


def test_corpus_is_reproducible(tiny_config: BacklineConfig) -> None:
    first = generate_corpus(tiny_config, n_songs=4)
    second = generate_corpus(tiny_config, n_songs=4)
    assert [s.truth_dict() for s in first] == [s.truth_dict() for s in second]
    assert [t.notes for s in first for t in s.score.tracks] == [
        t.notes for s in second for t in s.score.tracks
    ]


def test_song_does_not_depend_on_corpus_size(tiny_config: BacklineConfig) -> None:
    small = generate_corpus(tiny_config, n_songs=2)
    large = generate_corpus(tiny_config, n_songs=5)
    assert small[1].truth_dict() == large[1].truth_dict()


def test_other_seed_gives_other_songs(tiny_config: BacklineConfig) -> None:
    a = generate_corpus(tiny_config, n_songs=3, seed=1)
    b = generate_corpus(tiny_config, n_songs=3, seed=2)
    assert [s.truth_dict() for s in a] != [s.truth_dict() for s in b]


def test_every_song_has_a_band(corpus: list[SynthSong]) -> None:
    for song in corpus:
        roles = sorted(t.role for t in song.score.tracks)
        assert roles == sorted([TrackRole.GUITAR, TrackRole.BASS, TrackRole.DRUMS])
        end = song.score.end
        for track in song.score.tracks:
            assert track.notes, track.name
            assert all(n.end <= end for n in track.notes)


def test_bass_stays_in_its_instrument_range(
    corpus: list[SynthSong], tiny_config: BacklineConfig
) -> None:
    low, high = tiny_config.instruments.bass.train_range
    for song in corpus:
        bass = _track(song, TrackRole.BASS)
        assert bass.tuning is not None
        lowest, highest = min(bass.tuning), max(bass.tuning) + 24
        assert all(lowest <= n.pitch <= highest for n in bass.notes)
        assert all(low <= n.pitch <= high for n in bass.notes)


def test_drums_use_only_mapped_notes(corpus: list[SynthSong], tiny_config: BacklineConfig) -> None:
    for song in corpus:
        drums = _track(song, TrackRole.DRUMS)
        assert drums.is_percussion
        assert all(tiny_config.drums.class_of(n.pitch) is not None for n in drums.notes)


def test_chords_tile_the_song(corpus: list[SynthSong]) -> None:
    for song in corpus:
        chords = song.chords
        assert chords[0].start == 0
        assert chords[-1].end == song.score.end
        assert all(a.end == b.start for a, b in itertools.pairwise(chords))


def test_chord_roots_follow_section_keys(corpus: list[SynthSong]) -> None:
    for song in corpus:
        bar_ticks = song.score.time_signatures[0].bar_ticks()
        for chord in song.chords:
            bar = chord.start // bar_ticks
            section = next(s for s in song.sections if s.start_bar <= bar < s.start_bar + s.n_bars)
            assert chord.root == (section.tonic + chord.degree) % 12


def test_bass_truth_matches_bass_notes(corpus: list[SynthSong]) -> None:
    for song in corpus:
        bass = _track(song, TrackRole.BASS)
        assert Counter((n.start, n.pitch) for n in bass.notes) == Counter(
            (b.start, b.pitch) for b in song.bass_notes
        )


def test_bass_plays_the_root_on_chord_changes(corpus: list[SynthSong]) -> None:
    for song in corpus:
        by_start = {(b.start, b.pitch): b.role for b in song.bass_notes}
        starts = {b.start: b for b in song.bass_notes}
        for chord in song.chords:
            note = starts.get(swing_tick(chord.start) if song.swing else chord.start)
            if note is None:
                continue
            assert note.pitch % 12 == chord.root
            assert by_start[(note.start, note.pitch)] is BassRole.CHORD_TONE


def test_power_chords_keep_the_third_out_of_the_guitar(corpus: list[SynthSong]) -> None:
    checked = 0
    for song in corpus:
        guitar = _track(song, TrackRole.GUITAR)
        for chord in song.chords:
            if chord.quality is not ChordQuality.POWER:
                continue
            lo = swing_tick(chord.start) if song.swing else chord.start
            hi = swing_tick(chord.end) if song.swing else chord.end
            pcs = {(n.pitch - chord.root) % 12 for n in guitar.notes if lo <= n.start < hi}
            assert pcs <= {0, 7, 9}, (song.score.title, chord)
            checked += 1
    assert checked > 0


def test_corpus_covers_the_vocabulary(
    large_corpus: list[SynthSong], tiny_config: BacklineConfig
) -> None:
    roles = {b.role for s in large_corpus for b in s.bass_notes}
    assert roles == set(BassRole)
    drum_classes = {
        tiny_config.drums.class_of(n.pitch)
        for s in large_corpus
        for n in _track(s, TrackRole.DRUMS).notes
    }
    assert drum_classes == set(tiny_config.drums.classes)
    meters = {s.meter for s in large_corpus}
    assert meters == {"4/4", "3/4"}
    assert any(s.swing for s in large_corpus)
    assert any(len({sec.tonic for sec in s.sections}) > 1 for s in large_corpus)
    qualities = {c.quality for s in large_corpus for c in s.chords}
    assert {ChordQuality.POWER, ChordQuality.MAJOR, ChordQuality.MINOR} <= qualities


def test_three_four_songs_use_whole_bar_chords(large_corpus: list[SynthSong]) -> None:
    waltzes = [s for s in large_corpus if s.meter == "3/4"]
    assert waltzes
    for song in waltzes:
        bar_ticks = song.score.time_signatures[0].bar_ticks()
        assert bar_ticks == 2880
        assert all((c.end - c.start) % bar_ticks == 0 for c in song.chords)
