# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the corpus index from the imported library and calibrate its thresholds.

Calibration checks 8-bar windows of corpus songs against everything except their own song
group, which gives the distribution of similarity *between different songs*. The yellow
and red thresholds of the rare-pattern share are percentiles of that distribution. Two
more distributions are reported for orientation: against songs of the same artist, and
against other versions of the same song (which should look like copies).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from backline.config import BacklineConfig
from backline.formats.store import load_score
from backline.ingest.catalog import Catalog
from backline.ingest.identity import normalize_name
from backline.ingest.pipeline import LibraryPaths
from backline.originality.check import OriginalityChecker
from backline.originality.index import OriginalityIndex, SongEntry, build_channel
from backline.originality.represent import bass_events, degree_events, drum_cells
from backline.score import Score, TrackRole
from backline.theory.annotate import Annotation, annotate_score, load_annotation
from backline.theory.grid import beat_grid, beats_per_bar

ProgressCallback = Callable[[int, int], None]
WINDOW_BARS = 8
MIN_THRESHOLD = 0.02


def _tonic_lookup(annotation: Annotation) -> Callable[[int], int]:
    def tonic_at(tick: int) -> int:
        return annotation.key_at(tick).tonic

    return tonic_at


def slice_score(score: Score, start_bar: int, n_bars: int) -> Score:
    """The bars ``start_bar .. start_bar + n_bars`` as a score of their own."""
    starts = score.bar_starts()
    lo, hi = starts[start_bar], starts[start_bar + n_bars]
    tracks = [
        replace(
            track, notes=[replace(n, start=n.start - lo) for n in track.notes if lo <= n.start < hi]
        )
        for track in score.tracks
    ]
    return Score(
        n_bars=n_bars,
        time_signatures=[replace(score.time_signature_at(start_bar), bar=0)],
        tempos=[replace(score.tempos[0], tick=0, bpm=score.tempo_at(lo))],
        tracks=tracks,
        tpq=score.tpq,
        title=score.title,
        artist=score.artist,
    )


def build_originality_index(
    config: BacklineConfig,
    library: LibraryPaths,
    out_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> OriginalityIndex:
    """Index every imported song (dataset songs and their other versions), then calibrate."""
    cfg = config.originality
    with Catalog(library.catalog) as catalog:
        records = [s for s in catalog.songs() if s.status in ("ok", "duplicate")]
    group_ids: dict[str, int] = {}
    entries: list[SongEntry] = []
    bass, bass_bars, degree, degree_bars, drums, drum_bars = [], [], [], [], [], []
    per_bar: list[int] = []
    scores: dict[int, Score] = {}
    for done, record in enumerate(records, start=1):
        score = load_score(library.score_path(record.id))
        path = library.annotation_path(record.id)
        annotation = load_annotation(path) if path.is_file() else annotate_score(score)
        group = group_ids.setdefault(record.group_id or record.id, len(group_ids))
        entries.append(SongEntry(record.id, group, record.artist, record.title))
        line = score.tracks_with_role(TrackRole.BASS)[0].notes
        events = bass_events(line)
        bass.append(events.codes)
        bass_bars.append(np.array([score.bar_of(t) for t in events.starts], dtype=np.int32))
        steps = degree_events(line, _tonic_lookup(annotation))
        degree.append(steps.codes)
        degree_bars.append(np.array([score.bar_of(t) for t in steps.starts], dtype=np.int32))
        beats = beat_grid(score)
        drums.append(
            drum_cells(score.tracks_with_role(TrackRole.DRUMS)[0].notes, beats, config.drums)
        )
        drum_bars.append(np.array([b.bar for b in beats], dtype=np.int32))
        per_bar.append(beats_per_bar(score.time_signatures[0]))
        if record.status == "ok":
            scores[len(entries) - 1] = score
        if progress:
            progress(done, len(records))

    groups = [e.group for e in entries]
    index = OriginalityIndex(
        songs=entries,
        channels={
            "bass": build_channel(bass, bass_bars, groups, (cfg.seed, cfg.ngram)),
            "degree": build_channel(degree, degree_bars, groups, (cfg.ngram,)),
            "drums": build_channel(
                drums, drum_bars, groups, (cfg.drum_seed_beats, 2 * cfg.drum_seed_beats)
            ),
        },
        beats_per_bar=per_bar,
        meta={"config": config.name, "originality": cfg.model_dump(mode="json")},
    )
    index.meta.update(calibrate(index, scores, config))
    index.save(out_dir)
    return index


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {f"p{q}": round(float(np.percentile(values, q)), 4) for q in (50, 90, 95, 99)} | {
        "n": len(values)
    }


def calibrate(
    index: OriginalityIndex, scores: dict[int, Score], config: BacklineConfig
) -> dict[str, Any]:
    """Thresholds and reference distributions from 8-bar windows of corpus songs."""
    cfg = config.originality
    checker = OriginalityChecker(index, config)
    rng = random.Random(config.seed)
    sample = sorted(scores)
    rng.shuffle(sample)
    sample = sample[: cfg.calibration_songs]
    by_group: dict[int, list[int]] = defaultdict(list)
    by_artist: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(index.songs):
        by_group[entry.group].append(i)
        by_artist[normalize_name(entry.artist, artist=True)].append(i)

    between: dict[str, list[float]] = defaultdict(list)
    between_bars: dict[str, list[float]] = defaultdict(list)
    artist: dict[str, list[float]] = defaultdict(list)
    versions: dict[str, list[float]] = defaultdict(list)
    for song in sample:
        score = scores[song]
        if score.n_bars < WINDOW_BARS:
            window = score
        else:
            start = rng.randrange(score.n_bars - WINDOW_BARS + 1)
            window = slice_score(score, start, WINDOW_BARS)
        entry = index.songs[song]
        same_artist = [
            index.songs[i].song_id
            for i in by_artist[normalize_name(entry.artist, artist=True)]
            if index.songs[i].group != entry.group
        ]
        report = checker.check(window, references=same_artist, exclude_group=entry.group)
        for part in report.parts:
            between[part.part].append(part.corpus.rare_share)
            between_bars[part.part].append(part.corpus.longest_bars)
            if part.references is not None:
                artist[part.part].append(part.references.rare_share)
        others = [index.songs[i].song_id for i in by_group[entry.group] if i != song]
        if others:
            for part in checker.check(window, references=others).parts:
                if part.references is not None:
                    versions[part.part].append(part.references.rare_share)

    thresholds = {}
    for name in ("bass", "drums"):
        values = between.get(name, [])
        yellow = float(np.percentile(values, cfg.yellow_percentile)) if values else 0.15
        red = float(np.percentile(values, cfg.red_percentile)) if values else 0.35
        yellow = max(yellow, MIN_THRESHOLD)
        red = max(red, yellow + MIN_THRESHOLD)
        thresholds[name] = {"yellow": round(yellow, 4), "red": round(red, 4)}
    return {
        "thresholds": thresholds,
        "calibration": {
            part: {
                "between_songs": _percentiles(between.get(part, [])),
                "between_songs_longest_bars": _percentiles(between_bars.get(part, [])),
                "same_artist": _percentiles(artist.get(part, [])),
                "versions_of_a_song": _percentiles(versions.get(part, [])),
            }
            for part in ("bass", "drums")
        },
    }
