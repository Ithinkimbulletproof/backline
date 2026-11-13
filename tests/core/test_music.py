# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from backline.music import (
    ChordQuality,
    Mode,
    chord_symbol,
    degree_name,
    parse_pitch,
    pitch_name,
)


@pytest.mark.parametrize(
    ("name", "midi"),
    [
        ("E1", 28),
        ("B0", 23),
        ("C4", 60),
        ("F#2", 42),
        ("Bb1", 34),
        ("e2", 40),
        ("C-1", 0),
        (64, 64),
    ],
)
def test_parse_pitch(name: str | int, midi: int) -> None:
    assert parse_pitch(name) == midi


@pytest.mark.parametrize("bad", ["H2", "C#", "", "C-2", "G#9", 128, -1])
def test_parse_pitch_rejects_invalid(bad: str | int) -> None:
    with pytest.raises(ValueError, match="pitch"):
        parse_pitch(bad)


def test_parse_pitch_rejects_bool() -> None:
    with pytest.raises(TypeError):
        parse_pitch(True)


def test_pitch_name_round_trips_every_midi_note() -> None:
    for midi in range(128):
        assert parse_pitch(pitch_name(midi)) == midi
        assert parse_pitch(pitch_name(midi, flats=True)) == midi


def test_degree_names_cover_modal_degrees() -> None:
    assert degree_name(0) == "I"
    assert degree_name(3) == "bIII"
    assert degree_name(8) == "bVI"
    assert degree_name(10) == "bVII"
    assert degree_name(12 + 7) == "V"


def test_power_chord_has_no_third() -> None:
    assert not ChordQuality.POWER.has_third
    assert ChordQuality.POWER.third is None
    assert ChordQuality.MINOR.third == 3
    assert ChordQuality.DOMINANT7.third == 4
    assert ChordQuality.DOMINANT7.intervals == (0, 4, 7, 10)


def test_chord_symbols() -> None:
    assert chord_symbol(4, ChordQuality.POWER) == "E5"
    assert chord_symbol(9, ChordQuality.MINOR) == "Am"
    assert chord_symbol(9, ChordQuality.DOMINANT7) == "A7"
    assert chord_symbol(10, ChordQuality.MAJOR, flats=True) == "Bb"


def test_harmonic_functions() -> None:
    from backline.music import HarmonicFunction, function_label, harmonic_function

    assert harmonic_function(0) is HarmonicFunction.TONIC
    assert harmonic_function(9) is HarmonicFunction.TONIC
    assert harmonic_function(5) is HarmonicFunction.SUBDOMINANT
    assert harmonic_function(7) is HarmonicFunction.DOMINANT
    assert harmonic_function(10) is HarmonicFunction.MODAL
    assert harmonic_function(1) is HarmonicFunction.CHROMATIC
    assert [function_label(d) for d in (0, 5, 7, 10, 8, 3)] == [
        "T",
        "S",
        "D",
        "bVII",
        "bVI",
        "bIII",
    ]


def test_flat_spelling_follows_the_key() -> None:
    from backline.music import prefers_flats

    assert prefers_flats(5, Mode.MAJOR)  # F major
    assert prefers_flats(2, Mode.MINOR)  # D minor ~ F major
    assert prefers_flats(0, Mode.MIXOLYDIAN)  # C mixolydian ~ F major
    assert not prefers_flats(4, Mode.MINOR)  # E minor ~ G major
    assert not prefers_flats(9, Mode.DORIAN)  # A dorian ~ G major


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("E5", (4, ChordQuality.POWER)),
        ("Am", (9, ChordQuality.MINOR)),
        ("A7", (9, ChordQuality.DOMINANT7)),
        ("Bbm7", (10, ChordQuality.MINOR7)),
        ("F#", (6, ChordQuality.MAJOR)),
        ("cmaj7", (0, ChordQuality.MAJOR7)),
        ("Dsus4", (2, ChordQuality.SUS4)),
        ("Bdim", (11, ChordQuality.DIMINISHED)),
        (" G+ ", (7, ChordQuality.AUGMENTED)),
        ("N.C.", None),
        ("-", None),
    ],
)
def test_parse_chord_symbol(text: str, expected: tuple[int, ChordQuality] | None) -> None:
    from backline.music import parse_chord_symbol

    assert parse_chord_symbol(text) == expected


@pytest.mark.parametrize("bad", ["H", "E13", "Q5", "Am9b5"])
def test_parse_chord_symbol_rejects_unknown(bad: str) -> None:
    from backline.music import ChordSymbolError, parse_chord_symbol

    with pytest.raises(ChordSymbolError):
        parse_chord_symbol(bad)


def test_chord_symbols_round_trip() -> None:
    from backline.music import parse_chord_symbol

    for root in range(12):
        for quality in ChordQuality:
            assert parse_chord_symbol(chord_symbol(root, quality)) == (root, quality)
            assert parse_chord_symbol(chord_symbol(root, quality, flats=True)) == (root, quality)


def test_mode_scales() -> None:
    assert 10 in Mode.MIXOLYDIAN.scale
    assert 11 not in Mode.MIXOLYDIAN.scale
    assert Mode.MINOR.scale == (0, 2, 3, 5, 7, 8, 10)
