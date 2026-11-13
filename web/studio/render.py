# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Server-rendered SVG: the bass as a piano roll and the drums as a hit grid.

Bars that the originality check matched against the corpus are shaded, so a variant shows
at a glance where it resembles an existing song.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

from backline.score import Score, TrackRole

WIDTH = 640
ROLL_HEIGHT = 90
DRUM_ROWS = ("kick", "snare", "hats", "toms", "cymbals")
ROW_HEIGHT = 9


def piano_roll(score: Score, drum_groups: dict[int, str], highlight: Iterable[int] = ()) -> str:
    """SVG of the bass (top) and the drum hits by coarse group (bottom)."""
    end = max(score.end, 1)
    starts = score.bar_starts()
    height = ROLL_HEIGHT + len(DRUM_ROWS) * ROW_HEIGHT + 10

    def x(tick: int) -> float:
        return round(tick / end * WIDTH, 2)

    parts = [
        f'<svg class="roll" viewBox="0 0 {WIDTH} {height}" width="100%" role="img" '
        f'aria-label="Пиано-ролл баса и сетка барабанов">'
    ]
    for bar in sorted(set(highlight)):
        if 0 <= bar < score.n_bars:
            parts.append(
                f'<rect class="match" x="{x(starts[bar])}" y="0" '
                f'width="{x(starts[bar + 1]) - x(starts[bar])}" height="{height}"/>'
            )
    for tick in starts[1:-1]:
        parts.append(f'<line class="barline" x1="{x(tick)}" y1="0" x2="{x(tick)}" y2="{height}"/>')

    basses = score.tracks_with_role(TrackRole.BASS)
    notes = basses[0].notes if basses else []
    if notes:
        low = min(n.pitch for n in notes) - 2
        high = max(n.pitch for n in notes) + 2
        span = max(high - low, 1)
        for note in notes:
            y = round((high - note.pitch) / span * (ROLL_HEIGHT - 8), 2)
            width = max(x(note.end) - x(note.start) - 1, 1.5)
            parts.append(
                f'<rect class="bass" x="{x(note.start)}" y="{y}" width="{width}" height="6" rx="1">'
                f"<title>{escape(str(note.pitch))}</title></rect>"
            )
    drums = score.tracks_with_role(TrackRole.DRUMS)
    for note in drums[0].notes if drums else []:
        group = drum_groups.get(note.pitch)
        if group not in DRUM_ROWS:
            continue
        y = ROLL_HEIGHT + 4 + DRUM_ROWS.index(group) * ROW_HEIGHT
        parts.append(
            f'<rect class="hit {group}" x="{x(note.start)}" y="{y}" width="3" height="6" rx="1"/>'
        )
    parts.append("</svg>")
    return "".join(parts)
