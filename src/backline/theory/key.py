# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Key per bar with modulations: modal profiles, rock root priors and a Viterbi pass.

Every bar is scored against 48 candidate keys (12 tonics x major, minor, mixolydian,
dorian) using two kinds of evidence, each pooled over a few neighbouring bars:

* the pitch-class weights of guitar and bass against a modal profile. Major-family
  profiles tolerate the blues b3, b5 and b7, so blues licks do not flip a song to minor;
* the chord roots against root priors typical of rock (I-IV-V, bVII-IV-I, i-bVI-bVII...),
  which pins the tonic and the mode even when only power chords are played.

A Viterbi pass charges a penalty for every key change, so short borrowed chords do not
count as modulations. A span is marked ``mode_certain`` only if thirds actually sound in it.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from backline.music import Mode
from backline.theory.chords import ChordSpan
from backline.theory.grid import Beat

MODES: Final = (Mode.MAJOR, Mode.MINOR, Mode.MIXOLYDIAN, Mode.DORIAN)
CANDIDATES: Final = tuple((tonic, mode) for mode in MODES for tonic in range(12))

WINDOW: Final = 2
"""Bars on each side pooled into a bar's evidence."""
ROOT_WEIGHT: Final = 1.0
CHANGE_PENALTY: Final = 2.4
MIN_SPAN_BARS: Final = 4
THIRD_PRESENCE: Final = 0.04


def _pitch_profile(mode: Mode) -> npt.NDArray[np.float64]:
    profile = np.full(12, 0.05)
    third = 3 if mode in (Mode.MINOR, Mode.DORIAN) else 4
    for degree in mode.scale:
        profile[degree] = 0.45
    profile[0], profile[7], profile[third] = 1.0, 0.8, 0.7
    if mode in (Mode.MAJOR, Mode.MIXOLYDIAN):
        profile[3] = max(profile[3], 0.3)  # blue third
        profile[6] = max(profile[6], 0.2)  # blue fifth
        profile[10] = max(profile[10], 0.3)  # blue seventh
    return profile


_ROOT_PRIORS: Final = {
    Mode.MAJOR: {0: 1.0, 5: 0.7, 7: 0.7, 9: 0.4, 2: 0.3, 4: 0.2, 10: 0.25, 3: 0.2, 8: 0.15},
    Mode.MIXOLYDIAN: {0: 1.0, 10: 0.6, 5: 0.7, 7: 0.3, 2: 0.25, 9: 0.25},
    Mode.MINOR: {0: 1.0, 10: 0.6, 8: 0.6, 3: 0.5, 5: 0.55, 7: 0.35},
    Mode.DORIAN: {0: 1.0, 5: 0.65, 10: 0.5, 3: 0.4, 2: 0.3, 7: 0.3},
}


def _root_prior(mode: Mode) -> npt.NDArray[np.float64]:
    prior = np.full(12, 0.05)
    for degree, value in _ROOT_PRIORS[mode].items():
        prior[degree] = value
    return prior


def _rotations(make: Callable[[Mode], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    return np.array([np.roll(make(mode), tonic) for tonic, mode in CANDIDATES])


PITCH_PROFILES: Final = _rotations(_pitch_profile)
"""Candidate x pitch class."""
ROOT_PRIORS: Final = _rotations(_root_prior)

_MODE_PRIOR: Final = {Mode.MAJOR: 0.03, Mode.MINOR: 0.02, Mode.MIXOLYDIAN: 0.0, Mode.DORIAN: -0.01}
MODE_PRIOR: Final = np.array([_MODE_PRIOR[mode] for _, mode in CANDIDATES])
"""Without evidence for a modal degree, the plain major or minor reading wins."""

PHRASE_LENGTH: Final = 4
PHRASE_START_WEIGHT: Final = 3.0
"""Weight of the chord on the first beat of every 4-bar phrase, and of the song's first and
last chord: phrases tend to start on the tonic, and songs to start and end on it."""


@dataclass(frozen=True, slots=True)
class KeySpan:
    """A key holding over bars ``start_bar..end_bar`` (end exclusive)."""

    start_bar: int
    end_bar: int
    tonic: int
    mode: Mode
    mode_certain: bool


def bar_evidence(
    beats: Sequence[Beat],
    weights: npt.NDArray[np.float64],
    chords: Sequence[ChordSpan],
    n_bars: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-bar pitch weights (from beat weights) and duration-weighted chord roots."""
    pitches = np.zeros((n_bars, 12))
    for beat, row in zip(beats, weights, strict=True):
        pitches[beat.bar] += row
    roots = np.zeros((n_bars, 12))
    chord_starts = [c.start for c in chords]
    last = len(beats) - 1
    for position, beat in enumerate(beats):
        index = bisect.bisect_right(chord_starts, beat.start) - 1
        if index < 0 or chords[index].end <= beat.start or chords[index].root is None:
            continue
        phrase_start = beat.index == 0 and beat.bar % PHRASE_LENGTH == 0
        weight = PHRASE_START_WEIGHT if phrase_start or position in (0, last) else 1.0
        roots[beat.bar, chords[index].root] += weight
    return pitches, roots


def _pool(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Sum of each row with its ``window`` neighbours on both sides (clipped at the ends)."""
    n = len(values)
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    lo = np.clip(np.arange(n) - window, 0, n)
    hi = np.clip(np.arange(n) + window + 1, 0, n)
    pooled: npt.NDArray[np.float64] = cumulative[hi] - cumulative[lo]
    return pooled


def key_scores(
    pitches: npt.NDArray[np.float64],
    roots: npt.NDArray[np.float64],
    window: int | None = None,
) -> npt.NDArray[np.float64]:
    """Bar x candidate score from evidence pooled over neighbouring bars."""
    n_bars = len(pitches)
    width = WINDOW if window is None else window
    pooled_p = _pool(pitches, width)
    pooled_r = _pool(roots, width)
    scores = np.zeros((n_bars, len(CANDIDATES)))
    p_norm = np.linalg.norm(pooled_p, axis=1, keepdims=True)
    profile_norm = np.linalg.norm(PITCH_PROFILES, axis=1)
    has_p = p_norm[:, 0] > 0
    scores[has_p] += (pooled_p[has_p] / p_norm[has_p]) @ (PITCH_PROFILES / profile_norm[:, None]).T
    r_total = pooled_r.sum(axis=1, keepdims=True)
    has_r = r_total[:, 0] > 0
    scores[has_r] += ROOT_WEIGHT * (pooled_r[has_r] / r_total[has_r]) @ ROOT_PRIORS.T
    scores += MODE_PRIOR
    return scores


def _viterbi(scores: npt.NDArray[np.float64]) -> list[int]:
    dp = scores[0].copy()
    back = np.zeros(scores.shape, dtype=np.int64)
    states = np.arange(len(CANDIDATES))
    for t in range(1, len(scores)):
        best = int(np.argmax(dp))
        jump = dp[best] - CHANGE_PENALTY
        stay = dp >= jump
        back[t] = np.where(stay, states, best)
        dp = np.where(stay, dp, jump) + scores[t]
    path = [int(np.argmax(dp))]
    for t in range(len(scores) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def detect_keys(
    beats: Sequence[Beat],
    weights: npt.NDArray[np.float64],
    chords: Sequence[ChordSpan],
    n_bars: int,
) -> list[KeySpan]:
    """Key spans covering every bar."""
    pitches, roots = bar_evidence(beats, weights, chords, n_bars)
    path = _viterbi(key_scores(pitches, roots))
    path = _refine_boundaries(path, key_scores(pitches, roots, window=0))
    runs: list[list[int]] = []
    for bar, state in enumerate(path):
        if runs and runs[-1][0] == state:
            runs[-1][2] = bar + 1
        else:
            runs.append([state, bar, bar + 1])
    merged: list[list[int]] = []
    for run in runs:
        if merged and (run[2] - run[1] < MIN_SPAN_BARS or merged[-1][0] == run[0]):
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    if len(merged) > 1 and merged[0][2] - merged[0][1] < MIN_SPAN_BARS:
        merged[1][1] = merged[0][1]
        merged.pop(0)
    spans = []
    for state, start, end in merged:
        tonic, mode = CANDIDATES[state]
        spans.append(KeySpan(start, end, tonic, mode, _thirds_sound(beats, chords, start, end)))
    return spans


THIRD_SHARE: Final = 0.15
"""Share of a span's beats that must carry a chord with an audible third for its mode to
count as heard rather than inferred."""


def _thirds_sound(beats: Sequence[Beat], chords: Sequence[ChordSpan], start: int, end: int) -> bool:
    chord_starts = [c.start for c in chords]
    total = with_third = 0
    for beat in beats:
        if not start <= beat.bar < end:
            continue
        total += 1
        index = bisect.bisect_right(chord_starts, beat.start) - 1
        quality = chords[index].quality if index >= 0 else None
        with_third += bool(quality is not None and quality.has_third)
    return total > 0 and with_third / total >= THIRD_SHARE


BOUNDARY_SLACK: Final = 2
"""Bars a key change may move when it is aligned with the unpooled evidence."""


def _refine_boundaries(path: list[int], local: npt.NDArray[np.float64]) -> list[int]:
    """Move each key change to where the bar-level evidence switches best."""
    refined = list(path)
    changes = [t for t in range(1, len(path)) if path[t] != path[t - 1]]
    for t in changes:
        before, after = refined[t - 1], refined[t]
        lo, hi = max(t - BOUNDARY_SLACK, 1), min(t + BOUNDARY_SLACK, len(path) - 1)
        if any(refined[b] not in (before, after) for b in range(lo - 1, hi + 1)):
            continue
        best = max(
            range(lo, hi + 1),
            key=lambda c: local[lo - 1 : c, before].sum() + local[c : hi + 1, after].sum(),
        )
        for bar in range(lo, hi + 1):
            refined[bar] = before if bar < best else after
    return refined
