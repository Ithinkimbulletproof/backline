# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Annotate a score: chords per beat, keys per section, functions and bass note roles."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from backline.music import (
    BassRole,
    ChordQuality,
    HarmonicFunction,
    Mode,
    harmonic_function,
)
from backline.score import Note, Score, Track, TrackRole
from backline.theory.bass import bass_roles
from backline.theory.chords import ChordSpan, label_chords, pitch_weights
from backline.theory.grid import beat_grid
from backline.theory.key import KeySpan, detect_keys

FORMAT_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class LabelledChord:
    """A chord span read in its key."""

    start: int
    end: int
    root: int | None
    quality: ChordQuality | None
    tonic: int
    mode: Mode
    degree: int | None
    function: HarmonicFunction | None


@dataclass(slots=True)
class Annotation:
    """Everything the theory layer says about a song."""

    keys: list[KeySpan]
    chords: list[ChordSpan]
    bass_roles: list[BassRole] = field(default_factory=list)
    """One role per note of the bass track, in the bass track's sorted note order."""
    bar_starts: list[int] = field(default_factory=list)

    def key_at_bar(self, bar: int) -> KeySpan:
        index = bisect.bisect_right([k.start_bar for k in self.keys], bar) - 1
        return self.keys[max(index, 0)]

    def key_at(self, tick: int) -> KeySpan:
        bar = max(bisect.bisect_right(self.bar_starts, tick) - 1, 0)
        return self.key_at_bar(min(bar, len(self.bar_starts) - 2))

    def labelled_chords(self) -> list[LabelledChord]:
        labelled = []
        for chord in self.chords:
            key = self.key_at(chord.start)
            degree = None if chord.root is None else (chord.root - key.tonic) % 12
            labelled.append(
                LabelledChord(
                    chord.start,
                    chord.end,
                    chord.root,
                    chord.quality,
                    key.tonic,
                    key.mode,
                    degree,
                    None if degree is None else harmonic_function(degree),
                )
            )
        return labelled


def harmony_sources(score: Score) -> tuple[list[Note], Track | None]:
    """Guitar notes to read harmony from, and the bass track.

    The first guitar (the rhythm guitar in a stored song) is the main source; in bars where
    it is silent, the other guitars fill in.
    """
    guitars = score.tracks_with_role(TrackRole.GUITAR)
    basses = score.tracks_with_role(TrackRole.BASS)
    bass = basses[0] if basses else None
    if not guitars:
        return [], bass
    main = list(guitars[0].notes)
    covered = {score.bar_of(n.start) for n in main}
    for other in guitars[1:]:
        main.extend(n for n in other.notes if score.bar_of(n.start) not in covered)
    return main, bass


def annotate_score(score: Score) -> Annotation:
    """Analyse a score whose tracks carry roles (a stored song or a labelled riff)."""
    beats = beat_grid(score)
    guitar, bass = harmony_sources(score)
    bass_notes = bass.sorted_notes() if bass else []
    chords = label_chords(beats, guitar, bass_notes)
    weights, _ = pitch_weights(beats, guitar, bass_notes)
    keys = detect_keys(beats, weights, chords, score.n_bars)
    annotation = Annotation(keys=keys, chords=chords, bar_starts=score.bar_starts())
    annotation.bass_roles = bass_roles(bass_notes, chords, annotation.key_at)
    return annotation


# ------------------------------------------------------------------- storage


def annotation_to_dict(annotation: Annotation) -> dict[str, Any]:
    return {
        "version": FORMAT_VERSION,
        "bar_starts": annotation.bar_starts,
        "keys": [
            [k.start_bar, k.end_bar, k.tonic, k.mode.value, k.mode_certain] for k in annotation.keys
        ],
        "chords": [
            [c.start, c.end, c.root, c.quality.value if c.quality else None]
            for c in annotation.chords
        ],
        "bass_roles": [r.value for r in annotation.bass_roles],
    }


def annotation_from_dict(data: dict[str, Any]) -> Annotation:
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported annotation version: {data.get('version')!r}")
    return Annotation(
        keys=[KeySpan(s, e, t, Mode(m), bool(c)) for s, e, t, m, c in data["keys"]],
        chords=[
            ChordSpan(s, e, r, ChordQuality(q) if q is not None else None)
            for s, e, r, q in data["chords"]
        ],
        bass_roles=[BassRole(r) for r in data["bass_roles"]],
        bar_starts=list(data["bar_starts"]),
    )


def save_annotation(annotation: Annotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(annotation_to_dict(annotation), separators=(",", ":")), encoding="utf-8"
    )


def load_annotation(path: Path) -> Annotation:
    return annotation_from_dict(json.loads(path.read_text(encoding="utf-8")))
