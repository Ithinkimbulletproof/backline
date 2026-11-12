# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Helpers shared by format tests."""

from __future__ import annotations

from backline.config import VelocityConfig
from backline.score import Score


def note_signature(score: Score, velocity: VelocityConfig) -> list[list[tuple[object, ...]]]:
    """Per-track notes reduced to what must survive a round trip through a tab format.

    Loudness is compared as a dynamic level. Drum durations are left out: tabs notate a
    hit as lasting until the next one.
    """
    signature = []
    for track in score.tracks:
        rows: list[tuple[object, ...]] = []
        for note in track.notes:
            level = velocity.level_of(note.velocity, note.flags)
            if track.is_percussion:
                rows.append((note.start, note.pitch, level, int(note.flags)))
            else:
                rows.append((note.start, note.pitch, note.duration, level, int(note.flags)))
        signature.append(sorted(rows))
    return signature
