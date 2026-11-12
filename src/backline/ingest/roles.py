# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Decide what each track plays, and which guitar carries the riff.

Roles are voted from independent clues: the percussion channel, the instrument type a
Guitar Pro 7 file stores, the General MIDI program, the tuning, the track name and the
register of the notes. When the strongest clues disagree, or the vote is close, the track
is flagged as ambiguous so the import report can list it.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace

from backline.config import NameHints
from backline.music import pitch_name
from backline.score import Note, Score, Track, TrackRole

BASS_HIGHEST_OPEN = 48
"""A bass tuning's top string sits at or below C3 (G2 on a standard 4-string)."""
BASS_LOWEST_OPEN = 33
"""...and its bottom string at or below A1."""
GUITAR_HIGHEST_OPEN = 57
"""A guitar tuning's top string sits at or above A3 (E4 standard, B3 on a baritone)."""


@dataclass(frozen=True, slots=True)
class RoleDecision:
    """The role chosen for a track and why."""

    role: TrackRole
    confidence: float
    ambiguous: bool
    reasons: tuple[str, ...]


def _contains(name: str, hints: tuple[str, ...]) -> bool:
    return any(hint in name for hint in hints)


def classify_track(track: Track, hints: NameHints) -> RoleDecision:
    """Vote a role for one track."""
    votes: dict[TrackRole, float] = defaultdict(float)
    reasons: list[str] = []
    clues: set[TrackRole] = set()
    """Roles named by a structural clue (channel, instrument, program, tuning, name)."""

    def vote(role: TrackRole, weight: float, reason: str, *, is_strong: bool = True) -> None:
        votes[role] += weight
        reasons.append(f"{reason} -> {role.value}")
        if is_strong:
            clues.add(role)

    if track.is_percussion:
        vote(TrackRole.DRUMS, 10, "percussion channel", is_strong=True)

    instrument = track.instrument.lower()
    if instrument:
        if "drum" in instrument or "percussion" in instrument:
            vote(TrackRole.DRUMS, 6, f"instrument {track.instrument}", is_strong=True)
        elif "bass" in instrument:
            vote(TrackRole.BASS, 4, f"instrument {track.instrument}", is_strong=True)
        elif "guitar" in instrument:
            vote(TrackRole.GUITAR, 4, f"instrument {track.instrument}", is_strong=True)
        else:
            vote(TrackRole.OTHER, 3, f"instrument {track.instrument}", is_strong=True)

    if not track.is_percussion:
        if 32 <= track.program <= 39:
            vote(TrackRole.BASS, 2, f"program {track.program}")
        elif 24 <= track.program <= 31:
            vote(TrackRole.GUITAR, 2, f"program {track.program}")
        elif track.program >= 40:
            vote(TrackRole.OTHER, 1, f"program {track.program}")

    if track.tuning and not track.is_percussion:
        low, high, count = min(track.tuning), max(track.tuning), len(track.tuning)
        label = f"tuning {pitch_name(low)}..{pitch_name(high)} ({count} strings)"
        if high <= BASS_HIGHEST_OPEN and low <= BASS_LOWEST_OPEN and 4 <= count <= 6:
            vote(TrackRole.BASS, 3, label, is_strong=True)
        elif high >= GUITAR_HIGHEST_OPEN and 6 <= count <= 8:
            vote(TrackRole.GUITAR, 3, label, is_strong=True)

    name = track.name.lower()
    if _contains(name, hints.bass):
        vote(TrackRole.BASS, 2, f"name {track.name!r}")
    elif _contains(name, hints.guitar):
        vote(TrackRole.GUITAR, 2, f"name {track.name!r}")
    elif _contains(name, hints.drums):
        vote(TrackRole.DRUMS, 2, f"name {track.name!r}")

    if track.notes and not track.is_percussion:
        median = statistics.median(n.pitch for n in track.notes)
        label = f"median pitch {pitch_name(int(median))}"
        if median < 45:
            vote(TrackRole.BASS, 1, label, is_strong=False)
        elif median >= 52:
            vote(TrackRole.GUITAR, 1, label, is_strong=False)

    if not votes:
        return RoleDecision(TrackRole.OTHER, 0.0, True, ("no clues",))
    ranked = sorted(votes.items(), key=lambda item: -item[1])
    role, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    ambiguous = len(clues) > 1 or best - runner_up <= 1
    confidence = round(best / sum(votes.values()), 3)
    return RoleDecision(role, confidence, ambiguous, tuple(reasons))


# ------------------------------------------------------------------- guitars


@dataclass(frozen=True, slots=True)
class GuitarProfile:
    """How rhythm- or lead-like a guitar track is over a span of bars."""

    index: int
    notes: int
    chord_share: float
    """Share of onsets that strike two or more notes."""
    coverage: float
    """Share of bars in the span with at least one note."""
    median_pitch: float
    name_hint: int
    """+1 for a rhythm-guitar name, -1 for a lead or solo name."""
    lead_like: bool
    rhythm_score: float = field(compare=False)


def profile_guitar(
    index: int,
    track: Track,
    score: Score,
    hints: NameHints,
    bars: range | None = None,
) -> GuitarProfile:
    """Measure one guitar track, optionally only over ``bars``."""
    starts = score.bar_starts()
    span = bars if bars is not None else range(score.n_bars)
    lo, hi = starts[span.start], starts[span.stop]
    notes = [n for n in track.notes if lo <= n.start < hi]
    onsets: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        onsets[note.start].append(note)
    chord_share = sum(len(group) > 1 for group in onsets.values()) / len(onsets) if onsets else 0.0
    bars_with_notes = {score.bar_of(n.start) for n in notes}
    coverage = len(bars_with_notes) / max(len(span), 1)
    median = float(statistics.median(n.pitch for n in notes)) if notes else 0.0
    name = track.name.lower()
    hint = 1 if _contains(name, hints.rhythm) else -1 if _contains(name, hints.lead) else 0
    lead_like = hint < 0 or (bool(notes) and chord_share < 0.2 and median >= 62)
    rhythm_score = (
        2.0 * chord_share
        + coverage
        + (64.0 - median) / 12.0 * (1.0 if notes else 0.0)
        + 1.0 * hint
        - (2.0 if lead_like else 0.0)
    )
    return GuitarProfile(
        index=index,
        notes=len(notes),
        chord_share=round(chord_share, 3),
        coverage=round(coverage, 3),
        median_pitch=median,
        name_hint=hint,
        lead_like=lead_like,
        rhythm_score=round(rhythm_score, 3),
    )


def assign_roles(score: Score, hints: NameHints) -> Score:
    """Copy of ``score`` with voted roles and guitars ordered rhythm-first (nothing dropped).

    Used to analyse a single file, e.g. a riff with no bass or drums.
    """
    roles = [classify_track(track, hints).role for track in score.tracks]
    tracks = [replace(track, role=role) for track, role in zip(score.tracks, roles, strict=True)]
    guitars = [i for i, role in enumerate(roles) if role is TrackRole.GUITAR]
    relabelled = replace(score, tracks=tracks)
    ranked = [p.index for p in rank_guitars(relabelled, guitars, hints)]
    others = [i for i in range(len(tracks)) if i not in guitars]
    return replace(relabelled, tracks=[tracks[i] for i in [*ranked, *others]])


def rank_guitars(
    score: Score,
    indices: list[int],
    hints: NameHints,
    bars: range | None = None,
) -> list[GuitarProfile]:
    """Guitar tracks ordered from the most to the least rhythm-like (the riff carrier first)."""
    profiles = [profile_guitar(i, score.tracks[i], score, hints, bars) for i in indices]
    return sorted(profiles, key=lambda p: (-p.rhythm_score, p.index))
