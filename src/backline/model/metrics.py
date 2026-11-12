# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Quality metrics of a generated segment (besides originality).

* plan adherence: bass notes on strong beats whose pitch class is the root of the plan's
  chord there; shows whether the model follows the harmony it wrote;
* bass roles: shares of chord tones and passing notes, compared with the corpus;
* bass-kick coupling: Jaccard overlap of bass onsets and kick onsets;
* validity: every stream has the riff's bar count and every note fits its instrument.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass

from backline.config import BacklineConfig
from backline.music import BassRole
from backline.tokenizer.encode import Decoded

WHOLE_NOTE_STEPS = 96
"""Grid steps in a whole note (24 per quarter)."""


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    plan_adherence: float
    chord_tone_share: float
    passing_share: float
    bass_kick_coupling: float
    valid: bool


def _strong_positions(bar_steps: int, beats: int) -> set[int]:
    step = bar_steps // beats
    strong = {0}
    if beats % 2 == 0:
        strong.add((beats // 2) * step)
    return strong


def segment_metrics(decoded: Decoded, config: BacklineConfig, n_bars: int) -> SegmentMetrics:
    numerator, denominator = decoded.meter
    bar_steps = numerator * WHOLE_NOTE_STEPS // denominator
    beats = (
        numerator // 3 if denominator == 8 and numerator % 3 == 0 and numerator > 3 else numerator
    )
    strong = _strong_positions(bar_steps, beats)

    plan = sorted(decoded.plan, key=lambda c: (c.bar, c.pos))
    keys = [(c.bar, c.pos) for c in plan]
    hits = total = 0
    for note in decoded.bass:
        if note.pos not in strong:
            continue
        index = bisect.bisect_right(keys, (note.bar, note.pos)) - 1
        if index < 0 or plan[index].root is None:
            continue
        total += 1
        hits += note.pitch % 12 == plan[index].root
    adherence = hits / total if total else 0.0

    roles = Counter(n.role for n in decoded.bass if n.role is not None)
    counted = sum(roles.values())
    chord_share = roles[BassRole.CHORD_TONE] / counted if counted else 0.0
    passing_share = roles[BassRole.PASSING] / counted if counted else 0.0

    bass_onsets = {(n.bar, n.pos) for n in decoded.bass}
    kicks = {(h.bar, h.pos) for h in decoded.drums if h.drum_class == "kick"}
    union = bass_onsets | kicks
    coupling = len(bass_onsets & kicks) / len(union) if union else 0.0

    low, high = config.instruments.bass.active.lowest, config.instruments.bass.active.highest
    bars_ok = all(
        max((x.bar for x in stream), default=0) < n_bars
        for stream in (decoded.bass, decoded.drums, decoded.plan)
    )
    pitches_ok = all(low <= n.pitch <= high for n in decoded.bass)
    positions_ok = all(0 <= n.pos < bar_steps for n in decoded.bass) and all(
        0 <= h.pos < bar_steps for h in decoded.drums
    )
    plan_ok = all(any(c.bar == bar and c.pos == 0 for c in plan) for bar in range(n_bars))
    return SegmentMetrics(
        plan_adherence=round(adherence, 4),
        chord_tone_share=round(chord_share, 4),
        passing_share=round(passing_share, 4),
        bass_kick_coupling=round(coupling, 4),
        valid=bars_ok
        and pitches_ok
        and positions_ok
        and plan_ok
        and bool(decoded.bass or decoded.drums),
    )
