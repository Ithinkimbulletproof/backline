# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Music primitives shared by every stage: pitches, degrees, chord qualities, roles.

Pitch names use scientific pitch notation: C4 = MIDI 60, so the low E string of a
bass is E1 = 28 and the low B of a five-string bass is B0 = 23. Note that some DAWs
and plugins label octaves differently (FL Studio uses C5 = 60, Superior Drummer
uses C3 = 60); MIDI numbers are always authoritative.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

NOTE_NAMES_SHARP: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_NAMES_FLAT: Final = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_LETTER_PC: Final = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PITCH_RE: Final = re.compile(r"^([A-Ga-g])([#b]*)(-?\d+)$")

MIDI_MIN: Final = 0
MIDI_MAX: Final = 127


def parse_pitch(value: str | int) -> int:
    """Return the MIDI number for a pitch given as a name (``"E1"``, ``"F#2"``) or an int."""
    if isinstance(value, bool):
        raise TypeError("pitch must be a note name or a MIDI number, not a bool")
    if isinstance(value, int):
        pitch = value
    elif isinstance(value, str):
        match = _PITCH_RE.match(value.strip())
        if match is None:
            raise ValueError(f"invalid pitch name: {value!r}")
        letter, accidentals, octave = match.groups()
        shift = accidentals.count("#") - accidentals.count("b")
        pitch = (int(octave) + 1) * 12 + _LETTER_PC[letter.upper()] + shift
    else:
        raise TypeError(f"unsupported pitch value: {value!r}")
    if not MIDI_MIN <= pitch <= MIDI_MAX:
        raise ValueError(f"pitch {value!r} is outside the MIDI range 0..127")
    return pitch


def pitch_name(pitch: int, *, flats: bool = False) -> str:
    """Return the scientific name of a MIDI pitch, e.g. ``28 -> "E1"``."""
    if not MIDI_MIN <= pitch <= MIDI_MAX:
        raise ValueError(f"pitch {pitch} is outside the MIDI range 0..127")
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def pitch_class_name(pc: int, *, flats: bool = False) -> str:
    """Return the name of a pitch class 0..11 without an octave."""
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    return names[pc % 12]


DEGREE_NAMES: Final = (
    "I",
    "bII",
    "II",
    "bIII",
    "III",
    "IV",
    "#IV",
    "V",
    "bVI",
    "VI",
    "bVII",
    "VII",
)


def degree_name(semitones_above_tonic: int) -> str:
    """Return the Roman-numeral degree for a chromatic offset from the tonic (0..11)."""
    return DEGREE_NAMES[semitones_above_tonic % 12]


class Mode(StrEnum):
    """Modes the theory layer distinguishes; blues inflections live on top of these."""

    MAJOR = "major"
    MINOR = "minor"
    MIXOLYDIAN = "mixolydian"
    DORIAN = "dorian"

    @property
    def scale(self) -> tuple[int, ...]:
        """Scale degrees as semitones above the tonic."""
        return _MODE_SCALES[self]


_MODE_SCALES: Final = {
    Mode.MAJOR: (0, 2, 4, 5, 7, 9, 11),
    Mode.MINOR: (0, 2, 3, 5, 7, 8, 10),
    Mode.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
    Mode.DORIAN: (0, 2, 3, 5, 7, 9, 10),
}


class ChordQuality(StrEnum):
    """Chord qualities used in the harmonic plan.

    ``POWER`` is a root and a fifth with no audible third: the mode stays undecided.
    """

    POWER = "5"
    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    SUS2 = "sus2"
    SUS4 = "sus4"
    DOMINANT7 = "7"
    MAJOR7 = "maj7"
    MINOR7 = "m7"

    @property
    def intervals(self) -> tuple[int, ...]:
        """Chord tones as semitones above the root."""
        return _CHORD_INTERVALS[self]

    @property
    def has_third(self) -> bool:
        """Whether the quality implies an audible major or minor third."""
        return any(i in (3, 4) for i in self.intervals)

    @property
    def third(self) -> int | None:
        """Semitones from the root to the third, or ``None`` for thirdless chords."""
        for interval in self.intervals:
            if interval in (3, 4):
                return interval
        return None


_CHORD_INTERVALS: Final = {
    ChordQuality.POWER: (0, 7),
    ChordQuality.MAJOR: (0, 4, 7),
    ChordQuality.MINOR: (0, 3, 7),
    ChordQuality.DIMINISHED: (0, 3, 6),
    ChordQuality.AUGMENTED: (0, 4, 8),
    ChordQuality.SUS2: (0, 2, 7),
    ChordQuality.SUS4: (0, 5, 7),
    ChordQuality.DOMINANT7: (0, 4, 7, 10),
    ChordQuality.MAJOR7: (0, 4, 7, 11),
    ChordQuality.MINOR7: (0, 3, 7, 10),
}

_CHORD_SUFFIX: Final = {
    ChordQuality.POWER: "5",
    ChordQuality.MAJOR: "",
    ChordQuality.MINOR: "m",
    ChordQuality.DIMINISHED: "dim",
    ChordQuality.AUGMENTED: "aug",
    ChordQuality.SUS2: "sus2",
    ChordQuality.SUS4: "sus4",
    ChordQuality.DOMINANT7: "7",
    ChordQuality.MAJOR7: "maj7",
    ChordQuality.MINOR7: "m7",
}


def chord_symbol(root_pc: int, quality: ChordQuality, *, flats: bool = False) -> str:
    """Return a lead-sheet symbol such as ``"E5"``, ``"Am"`` or ``"A7"``."""
    return pitch_class_name(root_pc, flats=flats) + _CHORD_SUFFIX[quality]


class HarmonicFunction(StrEnum):
    """Function of a chord in its key.

    Rock harmony leans on modal degrees (bVII, bVI, bIII) that are neither tonic,
    subdominant nor dominant in the classical sense, so they get their own label.
    """

    TONIC = "T"
    SUBDOMINANT = "S"
    DOMINANT = "D"
    MODAL = "M"
    CHROMATIC = "X"


def harmonic_function(degree: int) -> HarmonicFunction:
    """Function of a chord whose root lies ``degree`` semitones above the tonic."""
    degree %= 12
    if degree in (0, 4, 9):
        return HarmonicFunction.TONIC
    if degree in (2, 5):
        return HarmonicFunction.SUBDOMINANT
    if degree in (7, 11):
        return HarmonicFunction.DOMINANT
    if degree in (3, 8, 10):
        return HarmonicFunction.MODAL
    return HarmonicFunction.CHROMATIC


def function_label(degree: int) -> str:
    """Short label for UIs: ``T``/``S``/``D``, the degree itself for modal and chromatic ones."""
    function = harmonic_function(degree)
    if function in (HarmonicFunction.MODAL, HarmonicFunction.CHROMATIC):
        return degree_name(degree)
    return function.value


FLAT_TONICS: Final = frozenset({5, 10, 3, 8, 1, 6})
"""Major-key tonics (F, Bb, Eb, Ab, Db, Gb) whose chords read better with flats."""


def prefers_flats(tonic: int, mode: Mode) -> bool:
    """Whether chord names in this key are conventionally spelled with flats."""
    offset = {Mode.MAJOR: 0, Mode.MINOR: 3, Mode.MIXOLYDIAN: 5, Mode.DORIAN: 10}[mode]
    return (tonic + offset) % 12 in FLAT_TONICS


class BassRole(StrEnum):
    """Harmonic role of a bass note relative to the chord it sounds against."""

    CHORD_TONE = "chord"
    PASSING = "passing"
    NEIGHBOR = "neighbor"
    CHROMATIC_APPROACH = "approach"
    BLUE_NOTE = "blue"


class Dynamic(StrEnum):
    """Quantized loudness levels, from softest to loudest."""

    GHOST = "ghost"
    NORMAL = "normal"
    ACCENT = "accent"
    MAX = "max"


DYNAMICS_ORDER: Final = (Dynamic.GHOST, Dynamic.NORMAL, Dynamic.ACCENT, Dynamic.MAX)
