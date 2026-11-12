# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Chord per beat: template matching over pitch weights, smoothed with a Viterbi pass.

Each beat gets a 12-bin pitch-class weight vector from the rhythm guitar and the bass.
Every (root, quality) template scores the beat: chord tones add, other notes subtract, and
a quality that implies a third, a seventh or a suspension is only allowed when that tone
actually sounds. A power chord therefore stays a power chord ("root + 5") and the mode is
never guessed. A bass note under the beat favours that root. The Viterbi pass rewards
staying on a chord so passing notes do not create one-beat chords.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from backline.music import ChordQuality
from backline.score import Note
from backline.theory.grid import Beat

QUALITIES: Final = tuple(ChordQuality)
N_STATES: Final = 12 * len(QUALITIES) + 1
NO_CHORD: Final = N_STATES - 1

_TONE_WEIGHT: Final = {
    0: 1.0,
    3: 1.0,
    4: 1.0,
    7: 0.8,
    6: 0.8,
    8: 0.8,
    10: 0.7,
    11: 0.7,
    2: 0.8,
    5: 0.8,
}
_COMPLEXITY: Final = {
    ChordQuality.POWER: 0.0,
    ChordQuality.MAJOR: 0.02,
    ChordQuality.MINOR: 0.02,
    ChordQuality.DOMINANT7: 0.08,
    ChordQuality.MINOR7: 0.08,
    ChordQuality.MAJOR7: 0.1,
    ChordQuality.SUS2: 0.1,
    ChordQuality.SUS4: 0.1,
    ChordQuality.DIMINISHED: 0.15,
    ChordQuality.AUGMENTED: 0.15,
}
NON_CHORD_PENALTY: Final = 0.6
PRESENCE: Final = 0.08
"""Share of a beat's weight a characteristic tone (third, seventh, sus) needs to count."""
BASS_ROOT_BONUS: Final = 0.25
GUITAR_ROOT_BONUS: Final = 0.25
"""Rock riffs are root-position power chords: the lowest guitar note marks the root."""
STAY_BONUS: Final = 0.3
ONSET_BOOST: Final = 1.25
GUITAR_WEIGHT: Final = 1.0
BASS_WEIGHT: Final = 0.8


def chord_tones(root: int, quality: ChordQuality) -> frozenset[int]:
    """Pitch classes of a chord."""
    return frozenset((root + i) % 12 for i in quality.intervals)


def _required(quality: ChordQuality) -> tuple[int, ...]:
    """Intervals that must sound for the quality to be claimed (beyond the root)."""
    return tuple(i for i in quality.intervals if i not in (0, 7))


def _templates() -> tuple[npt.NDArray[np.float64], list[tuple[int, ...]], npt.NDArray[np.float64]]:
    weights = np.full((N_STATES, 12), -NON_CHORD_PENALTY)
    required: list[tuple[int, ...]] = []
    complexity = np.zeros(N_STATES)
    for qi, quality in enumerate(QUALITIES):
        for root in range(12):
            state = qi * 12 + root
            for interval in quality.intervals:
                weights[state, (root + interval) % 12] = _TONE_WEIGHT[interval]
            required.append(tuple((root + i) % 12 for i in _required(quality)))
            complexity[state] = _COMPLEXITY[quality]
    weights[NO_CHORD] = 0.0
    required.append(())
    return weights, required, complexity


_WEIGHTS, _REQUIRED, _COMPLEXITY_VECTOR = _templates()


def state_label(state: int) -> tuple[int, ChordQuality] | None:
    """(root, quality) of a state, or ``None`` for "no chord"."""
    if state == NO_CHORD:
        return None
    return state % 12, QUALITIES[state // 12]


def pitch_weights(
    beats: Sequence[Beat],
    guitar: Iterable[Note],
    bass: Iterable[Note],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Per-beat pitch-class weights (rows sum to 1, or 0 for silence) and the bass under
    each beat (-1 when none)."""
    weights, bass_pc, _ = beat_features(beats, guitar, bass)
    return weights, bass_pc


def beat_features(
    beats: Sequence[Beat],
    guitar: Iterable[Note],
    bass: Iterable[Note],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Pitch weights, the bass under each beat and the lowest guitar note of each beat
    (pitch classes, -1 when none)."""
    starts = [b.start for b in beats]
    weights = np.zeros((len(beats), 12))
    bass_pc = np.full(len(beats), -1, dtype=np.int64)
    low_pitch = np.full(len(beats), 128, dtype=np.int64)

    def spread(note: Note, weight: float, track_bass: bool) -> None:
        first = max(bisect.bisect_right(starts, note.start) - 1, 0)
        for index in range(first, len(beats)):
            beat = beats[index]
            if beat.start >= note.end:
                break
            overlap = min(note.end, beat.end) - max(note.start, beat.start)
            if overlap <= 0:
                continue
            boost = ONSET_BOOST if beat.start <= note.start < beat.end else 1.0
            weights[index, note.pitch % 12] += weight * boost * overlap / (beat.end - beat.start)
            near_start = note.start <= beat.start + (beat.end - beat.start) // 4
            if track_bass and near_start and (bass_pc[index] < 0 or note.start <= beat.start):
                bass_pc[index] = note.pitch % 12
            if not track_bass:
                low_pitch[index] = min(low_pitch[index], note.pitch)

    for note in guitar:
        spread(note, GUITAR_WEIGHT, False)
    for note in sorted(bass, key=lambda n: n.start):
        spread(note, BASS_WEIGHT, True)
    totals = weights.sum(axis=1, keepdims=True)
    np.divide(weights, totals, out=weights, where=totals > 0)
    guitar_pc = np.where(low_pitch < 128, low_pitch % 12, -1)
    return weights, bass_pc, guitar_pc


def emissions(
    weights: npt.NDArray[np.float64],
    bass_pc: npt.NDArray[np.int64],
    guitar_pc: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float64]:
    """Score of every state for every beat."""
    scores = weights @ _WEIGHTS.T - _COMPLEXITY_VECTOR
    for state, tones in enumerate(_REQUIRED[:-1]):
        if tones:
            missing = (weights[:, list(tones)] < PRESENCE).any(axis=1)
            scores[missing, state] = -np.inf
    roots = np.arange(N_STATES - 1) % 12
    has_bass = bass_pc >= 0
    match = (roots[None, :] == bass_pc[:, None]) & has_bass[:, None]
    scores[:, :-1] += BASS_ROOT_BONUS * match
    if guitar_pc is not None:
        low = (roots[None, :] == guitar_pc[:, None]) & (guitar_pc >= 0)[:, None]
        scores[:, :-1] += GUITAR_ROOT_BONUS * low
    silent = weights.sum(axis=1) == 0
    scores[:, NO_CHORD] = np.where(silent, 0.05, -1.0)
    scores[silent, :-1] = np.minimum(scores[silent, :-1], -0.5)
    return scores


CHANGE_COST: Final = {"bar": 0.0, "half": 0.15, "beat": 0.45}
"""Extra cost of changing chord on a downbeat, mid-bar, or any other beat: rock harmony
moves with the bar and the half bar far more often than on beats 2 or 4."""


def change_costs(beats: Sequence[Beat]) -> npt.NDArray[np.float64]:
    """Cost of a chord change arriving on each beat."""
    per_bar: dict[int, int] = {}
    for beat in beats:
        per_bar[beat.bar] = per_bar.get(beat.bar, 0) + 1
    costs = []
    for beat in beats:
        count = per_bar[beat.bar]
        if beat.index == 0:
            costs.append(CHANGE_COST["bar"])
        elif count % 2 == 0 and beat.index == count // 2:
            costs.append(CHANGE_COST["half"])
        else:
            costs.append(CHANGE_COST["beat"])
    return np.array(costs)


def viterbi(
    scores: npt.NDArray[np.float64], costs: npt.NDArray[np.float64] | None = None
) -> list[int]:
    """Best state path: staying earns :data:`STAY_BONUS`, changing pays the beat's cost."""
    if len(scores) == 0:
        return []
    dp = scores[0].copy()
    back = np.zeros(scores.shape, dtype=np.int64)
    for t in range(1, len(scores)):
        best = int(np.argmax(dp))
        stay = dp + STAY_BONUS
        jump = dp[best] - (costs[t] if costs is not None else 0.0)
        take_stay = stay >= jump
        back[t] = np.where(take_stay, np.arange(N_STATES), best)
        dp = np.where(take_stay, stay, jump) + scores[t]
    path = [int(np.argmax(dp))]
    for t in range(len(scores) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


@dataclass(frozen=True, slots=True)
class ChordSpan:
    """A chord over ``start..end`` ticks; ``root``/``quality`` are ``None`` for no chord."""

    start: int
    end: int
    root: int | None
    quality: ChordQuality | None


def label_chords(
    beats: Sequence[Beat], guitar: Iterable[Note], bass: Iterable[Note]
) -> list[ChordSpan]:
    """Chord spans covering every beat, merged where the chord does not change."""
    if not beats:
        return []
    weights, bass_pc, guitar_pc = beat_features(beats, guitar, bass)
    path = viterbi(emissions(weights, bass_pc, guitar_pc), change_costs(beats))
    spans: list[ChordSpan] = []
    for beat, state in zip(beats, path, strict=True):
        label = state_label(state)
        root, quality = label if label else (None, None)
        if spans and spans[-1].root == root and spans[-1].quality == quality:
            last = spans[-1]
            spans[-1] = ChordSpan(last.start, beat.end, root, quality)
        else:
            spans.append(ChordSpan(beat.start, beat.end, root, quality))
    return spans
