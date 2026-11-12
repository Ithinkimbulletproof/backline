# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Cut a stored song into training segments, choosing the riff guitar per segment."""

from __future__ import annotations

from collections.abc import Iterator

from backline.config import BacklineConfig
from backline.ingest.roles import rank_guitars
from backline.score import Score, TrackRole
from backline.theory.annotate import Annotation
from backline.tokenizer.encode import SegmentSource


def _count(score: Score, track: int, bars: range) -> int:
    starts = score.bar_starts()
    lo, hi = starts[bars.start], starts[bars.stop]
    return sum(lo <= n.start < hi for n in score.tracks[track].notes)


def song_segments(
    score: Score, annotation: Annotation, config: BacklineConfig
) -> Iterator[SegmentSource]:
    """Segments of every configured length; each non-lead guitar may carry the riff.

    The most rhythm-like guitar of the segment comes first; with ``extra_guitars`` the other
    non-lead guitars yield extra segments with the same bass and drums.
    """
    tok = config.tokenizer
    guitars = [i for i, t in enumerate(score.tracks) if t.role is TrackRole.GUITAR]
    basses = [i for i, t in enumerate(score.tracks) if t.role is TrackRole.BASS]
    drums = [i for i, t in enumerate(score.tracks) if t.role is TrackRole.DRUMS]
    if not guitars or not basses or not drums:
        return
    bass, kit = basses[0], drums[0]
    for length in tok.segment_bars:
        stride = max(1, round(length * tok.stride_fraction))
        for start in range(0, score.n_bars - length + 1, stride):
            bars = range(start, start + length)
            if _count(score, bass, bars) < tok.min_notes_bass:
                continue
            if _count(score, kit, bars) < tok.min_notes_drums:
                continue
            ranked = [
                p
                for p in rank_guitars(score, guitars, config.ingest.name_hints, bars)
                if not p.lead_like and p.notes >= tok.min_notes_guitar
            ]
            chosen = ranked if tok.extra_guitars else ranked[:1]
            for profile in chosen:
                yield SegmentSource(score, annotation, profile.index, bass, kit, start, length)
