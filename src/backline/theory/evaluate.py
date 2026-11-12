# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Accuracy of the analysis against the ground truth of synthetic songs."""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from backline.music import ChordQuality, Mode
from backline.score import TrackRole
from backline.synth import SynthSong
from backline.synth.parts import BASS_STYLES_SOUNDING_THIRD
from backline.theory.annotate import Annotation, annotate_score
from backline.theory.grid import beat_grid


@dataclass(frozen=True, slots=True)
class TheoryScores:
    """Shares of correct labels over a corpus."""

    songs: int
    tonic: float
    """Bars whose tonic is right."""
    mode: float
    """Bars whose mode is right, counted only where the third audibly sounds."""
    root: float
    """Beats whose chord root is right."""
    third: float
    """Beats where "has a third" (vs power chord) is decided right."""
    quality: float
    """Beats whose exact chord quality is right."""
    bass_roles: float
    """Bass notes whose role is right."""

    def lines(self) -> list[str]:
        return [
            f"songs        {self.songs}",
            f"tonic        {self.tonic:.1%} of bars",
            f"mode         {self.mode:.1%} of bars with an audible third",
            f"chord root   {self.root:.1%} of beats",
            f"third        {self.third:.1%} of beats (third vs power chord)",
            f"quality      {self.quality:.1%} of beats",
            f"bass roles   {self.bass_roles:.1%} of notes",
        ]


def _same_mode(truth: Mode, found: Mode, blues: bool) -> bool:
    if blues:
        return found in (Mode.MAJOR, Mode.MIXOLYDIAN)
    return truth is found


def score_song(song: SynthSong, annotation: Annotation) -> Counter[str]:
    """Hit and total counts for one song."""
    counts: Counter[str] = Counter()
    score = song.score
    for section in song.sections:
        third_audible = section.full_chords or section.bass in BASS_STYLES_SOUNDING_THIRD
        blues = section.progression == "12-bar blues"
        for bar in range(section.start_bar, section.start_bar + section.n_bars):
            key = annotation.key_at_bar(bar)
            counts["tonic_total"] += 1
            counts["tonic_hit"] += key.tonic == section.tonic
            if third_audible:
                counts["mode_total"] += 1
                counts["mode_hit"] += key.tonic == section.tonic and _same_mode(
                    section.mode, key.mode, blues
                )

    truth_starts = [c.start for c in song.chords]
    found_starts = [c.start for c in annotation.chords]
    for beat in beat_grid(score):
        truth = song.chords[bisect.bisect_right(truth_starts, beat.start) - 1]
        found = annotation.chords[bisect.bisect_right(found_starts, beat.start) - 1]
        counts["beats"] += 1
        counts["root_hit"] += found.root == truth.root
        truth_third = truth.quality is not ChordQuality.POWER
        found_third = found.quality is not None and found.quality.has_third
        counts["third_hit"] += truth_third == found_third
        counts["quality_hit"] += found.quality == truth.quality

    bass = score.tracks_with_role(TrackRole.BASS)[0].sorted_notes()
    truth_roles = {(b.start, b.pitch): b.role for b in song.bass_notes}
    for note, role in zip(bass, annotation.bass_roles, strict=True):
        counts["bass_total"] += 1
        counts["bass_hit"] += truth_roles.get((note.start, note.pitch)) == role
    return counts


def evaluate(songs: Iterable[SynthSong]) -> TheoryScores:
    """Analyse every song and compare with its ground truth."""
    totals: Counter[str] = Counter()
    n = 0
    for song in songs:
        totals.update(score_song(song, annotate_score(song.score)))
        n += 1

    def share(hit: str, total: str) -> float:
        return totals[hit] / totals[total] if totals[total] else 0.0

    return TheoryScores(
        songs=n,
        tonic=share("tonic_hit", "tonic_total"),
        mode=share("mode_hit", "mode_total"),
        root=share("root_hit", "beats"),
        third=share("third_hit", "beats"),
        quality=share("quality_hit", "beats"),
        bass_roles=share("bass_hit", "bass_total"),
    )
