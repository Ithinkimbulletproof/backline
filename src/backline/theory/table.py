# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Plain-text view of an annotation: one row per bar, one column per beat."""

from __future__ import annotations

import bisect

from backline.music import Mode, chord_symbol, function_label, pitch_class_name, prefers_flats
from backline.score import Score
from backline.theory.annotate import Annotation
from backline.theory.grid import beat_grid
from backline.theory.key import KeySpan

_MODE_SHORT = {Mode.MAJOR: "", Mode.MINOR: "m", Mode.MIXOLYDIAN: " mix", Mode.DORIAN: " dor"}


def key_name(key: KeySpan, *, short: bool = False) -> str:
    """``E minor`` / ``Em``; an inferred mode (no thirds heard) is marked with ``?``."""
    flats = prefers_flats(key.tonic, key.mode)
    tonic = pitch_class_name(key.tonic, flats=flats)
    if short:
        return tonic + _MODE_SHORT[key.mode] + ("" if key.mode_certain else "?")
    suffix = "" if key.mode_certain else " (mode inferred: no thirds heard)"
    return f"{tonic} {key.mode.value}{suffix}"


def annotation_table(score: Score, annotation: Annotation, bars: range | None = None) -> list[str]:
    """Keys, then a row per bar: key, chord per beat ("." repeats), function per beat."""
    lines = [f"bars {k.start_bar + 1}-{k.end_bar}: {key_name(k)}" for k in annotation.keys]
    selected = bars if bars is not None else range(score.n_bars)
    starts = [c.start for c in annotation.chords]
    labelled = annotation.labelled_chords()
    beats = beat_grid(score)
    by_bar: dict[int, list[str]] = {}
    functions: dict[int, list[str]] = {}
    previous = None
    for beat in beats:
        if beat.bar not in selected:
            continue
        index = bisect.bisect_right(starts, beat.start) - 1
        chord = labelled[index]
        if chord.root is None or chord.quality is None:
            symbol, function = "N.C.", "-"
        else:
            symbol = chord_symbol(
                chord.root, chord.quality, flats=prefers_flats(chord.tonic, chord.mode)
            )
            function = function_label(chord.degree or 0)
        same = previous == index and beat.index > 0
        by_bar.setdefault(beat.bar, []).append("." if same else symbol)
        functions.setdefault(beat.bar, []).append("." if same else function)
        previous = index
    lines.append(f"{'bar':>4}  {'key':<8} {'chords':<32} functions")
    for bar in selected:
        key = key_name(annotation.key_at_bar(bar), short=True)
        chords = " ".join(f"{c:<6}" for c in by_bar.get(bar, []))
        funcs = " ".join(f"{f:<4}" for f in functions.get(bar, []))
        lines.append(f"{bar + 1:>4}  {key:<8} {chords:<32} {funcs}".rstrip())
    return lines
