# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Output drum maps: how each drum class is played on a concrete drum instrument.

A map assigns exactly one MIDI note to every drum class. A class the kit does not
have is marked ``substitute: true`` and borrows another sound; :meth:`DrumMap.substitutions`
reports those so the UI can warn the user.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backline.music import Dynamic

GM_PERCUSSION_NAMES: dict[int, str] = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi-Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}
"""General MIDI level 1 percussion names (channel 10)."""


class DrumMapError(ValueError):
    """Raised when a drum map file is malformed or does not cover the configured classes."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DrumMapEntry(_Frozen):
    """The note that plays one drum class."""

    note: int = Field(ge=0, le=127)
    substitute: bool = False


class RimshotOption(_Frozen):
    """Play snare hits at the loudest level as a rimshot articulation."""

    enabled: bool = False
    note: int = Field(ge=0, le=127)


class DrumMapOptions(_Frozen):
    """Optional instrument-specific behaviour."""

    snare_max_velocity_to_rimshot: RimshotOption | None = None


@dataclass(frozen=True, slots=True)
class Substitution:
    """A class missing from the kit and the sound used in its place."""

    missing: str
    played_as: str | None
    note: int


class DrumMap(_Frozen):
    """Class-to-note map for one drum instrument or preset."""

    name: str
    description: str = ""
    classes: dict[str, DrumMapEntry]
    options: DrumMapOptions = DrumMapOptions()

    def note_for(self, drum_class: str, dynamic: Dynamic = Dynamic.NORMAL) -> int:
        """MIDI note that plays ``drum_class`` at the given dynamic level."""
        try:
            entry = self.classes[drum_class]
        except KeyError:
            raise DrumMapError(f"drum map {self.name!r} has no class {drum_class!r}") from None
        rimshot = self.options.snare_max_velocity_to_rimshot
        if drum_class == "snare" and dynamic is Dynamic.MAX and rimshot and rimshot.enabled:
            return rimshot.note
        return entry.note

    def substitutions(self) -> list[Substitution]:
        """Classes the kit lacks, with the class whose sound replaces each of them."""
        owners = {entry.note: name for name, entry in self.classes.items() if not entry.substitute}
        return [
            Substitution(missing=name, played_as=owners.get(entry.note), note=entry.note)
            for name, entry in self.classes.items()
            if entry.substitute
        ]

    def check_covers(self, classes: Iterable[str]) -> None:
        """Raise :class:`DrumMapError` unless the map defines exactly ``classes``."""
        expected = set(classes)
        defined = set(self.classes)
        missing = sorted(expected - defined)
        extra = sorted(defined - expected)
        problems = []
        if missing:
            problems.append(f"missing classes: {', '.join(missing)}")
        if extra:
            problems.append(f"unknown classes: {', '.join(extra)}")
        if problems:
            raise DrumMapError(f"drum map {self.name!r}: " + "; ".join(problems))


def load_drum_map(path: Path) -> DrumMap:
    """Read and validate a drum map YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DrumMapError(f"cannot read drum map {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DrumMapError(f"drum map {path} is not valid YAML: {exc}") from exc
    try:
        return DrumMap.model_validate(raw)
    except ValidationError as exc:
        raise DrumMapError(f"drum map {path} is invalid:\n{exc}") from exc
