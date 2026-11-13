# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Check a bass line and a drum part against the corpus index.

Matching is seed-and-extend: every exact seed shared with a corpus song proposes an
alignment; the alignment grows in both directions while no window of ``ngram`` events
holds more than ``max_mismatches`` differences. Seeds that occur in many song groups are
common vocabulary ("root eighths") and are not searched at all.

The report gives the longest copied fragment from one song, the rarity-weighted share of
the part's n-grams found in the corpus, the closest sources, the same against the chosen
references, a score from 0 to 100 and a verdict. A continuous copy of ``red_bars`` or
more from one song is red whatever the score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from backline.config import BacklineConfig
from backline.originality.index import Channel, OriginalityIndex, ngram_hashes
from backline.originality.represent import (
    bass_events,
    beat_jaccard,
    drum_cells,
    rescale,
)
from backline.score import Score, TrackRole
from backline.theory.grid import beat_grid, beats_per_bar

DEFAULT_THRESHOLDS: Final = {"yellow": 0.15, "red": 0.35}
VERDICTS: Final = ("green", "yellow", "red")


@dataclass(frozen=True, slots=True)
class Fragment:
    """A stretch of the checked part that matches a stretch of a corpus song (event indices)."""

    song: int
    query_start: int
    query_end: int
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class SourceMatch:
    """A corpus song the part resembles, with the longest shared fragment."""

    song_id: str
    artist: str
    title: str
    bars: float
    query_bars: tuple[int, int]
    source_bars: tuple[int, int]
    """1-based, inclusive."""


@dataclass(slots=True)
class Similarity:
    """How much of a part resembles a set of songs, and which ones."""

    longest_bars: float = 0.0
    rare_share: float = 0.0
    sources: list[SourceMatch] = field(default_factory=list)
    matched_bars: list[int] = field(default_factory=list)
    """0-based bars of the checked part covered by a copied fragment."""


@dataclass(slots=True)
class PartReport:
    """Verdict for one part (bass or drums) against the corpus and the references."""

    part: str
    corpus: Similarity
    references: Similarity | None
    score: int
    verdict: str
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OriginalityReport:
    """Verdict for a generated variant: the worst of its parts."""

    parts: list[PartReport]
    score: int
    verdict: str

    @property
    def matched_bars(self) -> dict[str, list[int]]:
        return {p.part: p.corpus.matched_bars for p in self.parts}


# ------------------------------------------------------------------ matching


def _extend(
    equal: npt.NDArray[np.bool_], start: int, end: int, window: int, budget: int
) -> tuple[int, int]:
    """Grow ``[start, end)`` while every window inside keeps at most ``budget`` misses."""
    misses = np.concatenate([[0], np.cumsum(~equal)])
    n = len(equal)

    def window_ok(lo: int, hi: int) -> bool:
        return int(misses[hi] - misses[lo]) <= budget

    left, right = start, end
    while right < n and window_ok(max(left, right + 1 - window), right + 1):
        right += 1
    while left > 0 and window_ok(left - 1, min(right, left - 1 + window)):
        left -= 1
    while left < right and not equal[left]:
        left += 1
    while right > left and not equal[right - 1]:
        right -= 1
    return left, right


def find_fragments(
    query: npt.NDArray[np.int64],
    channel: Channel,
    *,
    seed: int,
    window: int,
    budget: int,
    max_groups: int,
    allowed: Callable[[int], bool],
    same: Callable[[npt.NDArray[np.int64], npt.NDArray[np.int64]], npt.NDArray[np.bool_]]
    | None = None,
    skip: npt.NDArray[np.bool_] | None = None,
) -> list[Fragment]:
    """Fragments shared with corpus songs (at least ``seed`` events long)."""
    postings = channel.postings[seed]
    hashes = ngram_hashes(query, seed)
    frequency = postings.group_frequency(hashes)
    diagonals: dict[tuple[int, int], list[int]] = {}
    for position, (value, groups) in enumerate(zip(hashes, frequency, strict=True)):
        if groups == 0 or groups > max_groups or (skip is not None and skip[position]):
            continue
        songs, starts = postings.lookup(value)
        for song, start in zip(songs.tolist(), starts.tolist(), strict=True):
            if allowed(song):
                diagonals.setdefault((song, start - position), []).append(position)
    compare = same or (lambda a, b: a == b)
    fragments: list[Fragment] = []
    for (song, offset), positions in diagonals.items():
        source = channel.song_codes(song)
        lo = max(0, -offset)
        hi = min(len(query), len(source) - offset)
        equal = compare(query[lo:hi], source[lo + offset : hi + offset])
        covered = -1
        for position in sorted(positions):
            if position < covered:
                continue
            a, b = _extend(equal, position - lo, position - lo + seed, window, budget)
            covered = b + lo
            if b - a >= seed:
                fragments.append(Fragment(song, a + lo, b + lo, a + lo + offset, b + lo + offset))
    return fragments


def rare_share(
    query: npt.NDArray[np.int64],
    channel: Channel,
    n: int,
    index: OriginalityIndex,
    exclude: set[int] | None = None,
    only: set[int] | None = None,
) -> float:
    """Rarity-weighted share of the query's n-grams that occur in the corpus.

    ``exclude`` lists hashes whose group frequency drops by one (the query's own group
    when calibrating on corpus songs); ``only`` restricts matches to hashes found in a
    subset of songs (the references).
    """
    hashes = ngram_hashes(query, n)
    if len(hashes) == 0:
        return 0.0
    frequency = channel.postings[n].group_frequency(hashes)
    values = hashes.tolist()
    if exclude:
        own = np.fromiter((h in exclude for h in values), dtype=bool, count=len(hashes))
        frequency = frequency - own
    if only is not None:
        keep = np.fromiter((h in only for h in values), dtype=bool, count=len(hashes))
        frequency = np.where(keep, frequency, 0)
    weights = index.idf(frequency)
    return float(weights.sum() / (len(hashes) * index.max_idf())) if index.max_idf() > 0 else 0.0


def _similar_beats(
    threshold: float,
) -> Callable[[npt.NDArray[np.int64], npt.NDArray[np.int64]], npt.NDArray[np.bool_]]:
    """Beat-by-beat comparison: Jaccard similarity of onsets at or above ``threshold``."""

    def similar(a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        return np.array(
            [beat_jaccard(int(x), int(y)) >= threshold for x, y in zip(a, b, strict=True)],
            dtype=bool,
        )

    return similar


# ------------------------------------------------------------------ checker


class OriginalityChecker:
    """Checks the bass and drum tracks of a score against an index."""

    def __init__(self, index: OriginalityIndex, config: BacklineConfig) -> None:
        self.index = index
        self.config = config
        self.cfg = config.originality
        self.song_index = {s.song_id: i for i, s in enumerate(index.songs)}
        self.max_groups = max(1, int(self.cfg.common_seed_share * index.n_groups))
        self.thresholds = index.meta.get("thresholds", {})

    def check(
        self,
        score: Score,
        *,
        references: Sequence[str] = (),
        exclude_group: int | None = None,
    ) -> OriginalityReport:
        """Report on ``score``'s bass and drums (tracks with those roles)."""
        reference_songs = {self.song_index[s] for s in references if s in self.song_index}
        parts = []
        basses = score.tracks_with_role(TrackRole.BASS)
        if basses:
            parts.append(self._part("bass", score, basses[0].notes, reference_songs, exclude_group))
        drums = score.tracks_with_role(TrackRole.DRUMS)
        if drums:
            parts.append(self._part("drums", score, drums[0].notes, reference_songs, exclude_group))
        if not parts:
            return OriginalityReport([], 100, "green")
        worst = max(parts, key=lambda p: VERDICTS.index(p.verdict))
        return OriginalityReport(parts, min(p.score for p in parts), worst.verdict)

    def _part(
        self,
        part: str,
        score: Score,
        notes: Sequence[object],
        references: set[int],
        exclude_group: int | None,
    ) -> PartReport:
        def allow_all(song: int) -> bool:
            return exclude_group is None or self.index.songs[song].group != exclude_group

        corpus = self._similarity(part, score, notes, allow_all, exclude_group, None)
        reference = None
        if references:

            def allow_refs(song: int) -> bool:
                return song in references and allow_all(song)

            reference = self._similarity(part, score, notes, allow_refs, exclude_group, references)
        limits = self.thresholds.get(part, DEFAULT_THRESHOLDS)
        reasons = []
        verdict = "green"
        worst_bars = max(corpus.longest_bars, reference.longest_bars if reference else 0.0)
        worst_share = max(corpus.rare_share, reference.rare_share if reference else 0.0)
        if worst_bars >= self.cfg.red_bars:
            verdict = "red"
            reasons.append(f"{worst_bars:.1f} bars copied from one song")
        if worst_share >= limits["red"]:
            verdict = "red"
            reasons.append(f"rare-pattern share {worst_share:.0%} is at duplicate level")
        if verdict != "red":
            if worst_bars >= self.cfg.yellow_bars:
                verdict = "yellow"
                reasons.append(f"{worst_bars:.1f} bars shared with one song")
            if worst_share >= limits["yellow"]:
                verdict = "yellow"
                reasons.append(f"rare-pattern share {worst_share:.0%} is unusually high")
        pressure = max(worst_bars / self.cfg.red_bars, worst_share / max(limits["red"], 1e-9))
        score_value = round(100 * (1 - min(max(pressure, 0.0), 1.0)))
        if verdict == "red":
            score_value = min(score_value, 33)
        return PartReport(part, corpus, reference, score_value, verdict, reasons)

    def _similarity(
        self,
        part: str,
        score: Score,
        notes: Sequence[object],
        allowed: Callable[[int], bool],
        exclude_group: int | None,
        subset: set[int] | None,
    ) -> Similarity:
        starts = score.bar_starts()
        if part == "bass":
            sequence = bass_events(notes)  # type: ignore[arg-type]
            channel = self.index.channels["bass"]
            fragments = find_fragments(
                sequence.codes,
                channel,
                seed=self.cfg.seed,
                window=self.cfg.ngram,
                budget=self.cfg.max_mismatches,
                max_groups=self.max_groups,
                allowed=allowed,
            )
            spans = [
                (
                    f,
                    sequence.starts[f.query_start],
                    sequence.ends[f.query_end - 1],
                )
                for f in fragments
            ]
            exclude = self._own_hashes(channel, exclude_group, self.cfg.ngram)
            only = self._subset_hashes(channel, subset, self.cfg.ngram)
            share = rare_share(sequence.codes, channel, self.cfg.ngram, self.index, exclude, only)
            measured = [
                (
                    f,
                    (end - start) / (starts[1] - starts[0]),
                    score.bar_of(start),
                    score.bar_of(min(end, starts[-1]) - 1),
                )
                for f, start, end in spans
            ]
        else:
            beats = beat_grid(score)
            cells = drum_cells(notes, beats, self.config.drums)  # type: ignore[arg-type]
            channel = self.index.channels["drums"]
            groups = len(self.config.drums.coarse)
            per_bar = beats_per_bar(score.time_signatures[0])
            measured = []
            for factor in (1.0, 2.0, 0.5):
                query = cells if factor == 1.0 else rescale(cells, factor, groups)
                window = 2 * self.cfg.drum_seed_beats
                empty = (
                    np.convolve(
                        (query == 0).astype(int), np.ones(self.cfg.drum_seed_beats, int), "valid"
                    )
                    == self.cfg.drum_seed_beats
                )
                threshold = self.cfg.drum_beat_similarity
                fragments = find_fragments(
                    query,
                    channel,
                    seed=self.cfg.drum_seed_beats,
                    window=window,
                    budget=self.cfg.max_mismatches,
                    max_groups=self.max_groups,
                    allowed=allowed,
                    same=_similar_beats(threshold),
                    skip=empty,
                )
                for f in fragments:
                    first = int(f.query_start / factor)
                    last = max(int((f.query_end - 1) / factor), first)
                    first_bar = beats[min(first, len(beats) - 1)].bar
                    last_bar = beats[min(last, len(beats) - 1)].bar
                    measured.append(
                        (f, (f.query_end - f.query_start) / factor / per_bar, first_bar, last_bar)
                    )
            n = 2 * self.cfg.drum_seed_beats
            exclude = self._own_hashes(channel, exclude_group, n)
            only = self._subset_hashes(channel, subset, n)
            share = rare_share(cells, channel, n, self.index, exclude, only)
        return self._summarise(channel, measured, share)

    def _own_hashes(self, channel: Channel, group: int | None, n: int) -> set[int] | None:
        if group is None:
            return None
        songs = {i for i, entry in enumerate(self.index.songs) if entry.group == group}
        return self._subset_hashes(channel, songs, n)

    @staticmethod
    def _subset_hashes(channel: Channel, songs: set[int] | None, n: int) -> set[int] | None:
        if songs is None:
            return None
        hashes: set[int] = set()
        for song in songs:
            hashes.update(ngram_hashes(channel.song_codes(song), n).tolist())
        return hashes

    def _summarise(
        self, channel: Channel, measured: list[tuple[Fragment, float, int, int]], share: float
    ) -> Similarity:
        best: dict[int, tuple[Fragment, float, int, int]] = {}
        covered: set[int] = set()
        for item in measured:
            fragment, bars, first, last = item
            covered.update(range(first, last + 1))
            if fragment.song not in best or bars > best[fragment.song][1]:
                best[fragment.song] = item
        sources = []
        for song, (fragment, bars, first, last) in sorted(best.items(), key=lambda kv: -kv[1][1])[
            :5
        ]:
            entry = self.index.songs[song]
            source_bars = channel.song_bars(song)
            s_first = int(source_bars[min(fragment.source_start, len(source_bars) - 1)])
            s_last = int(source_bars[min(fragment.source_end - 1, len(source_bars) - 1)])
            sources.append(
                SourceMatch(
                    entry.song_id,
                    entry.artist,
                    entry.title,
                    round(bars, 2),
                    (first + 1, last + 1),
                    (s_first + 1, s_last + 1),
                )
            )
        longest = sources[0].bars if sources else 0.0
        return Similarity(longest, round(share, 4), sources, sorted(covered))
