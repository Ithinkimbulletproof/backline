# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Import a tab library into the catalog.

Every tab file is read, its tracks get roles, songs without a guitar, a bass and drums are
filtered out, versions of the same song are grouped, one canonical version per group is
kept, and groups are split into train/val/test. Each kept song is stored as a normalised
score: guitars first (the most rhythm-like one leading), then the bass, then all drum
tracks merged into one. A JSON report lists everything that was skipped and why.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backline.config import BacklineConfig, PathSettings
from backline.formats.common import ReadStats, ScoreReadError, UnsupportedFormatError
from backline.formats.reader import KNOWN_UNSUPPORTED, SUPPORTED_SUFFIXES, read_score
from backline.formats.store import load_score, save_score
from backline.ingest.catalog import Catalog, SongRecord, TrackRecord
from backline.ingest.identity import (
    SongKeys,
    assign_split,
    content_key,
    file_digest,
    group_songs,
    name_key,
    resolve_metadata,
)
from backline.ingest.roles import RoleDecision, classify_track, rank_guitars
from backline.score import Score, Track, TrackRole

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""Called with (files done, files total) while files are being read."""

PARALLEL_THRESHOLD = 24
"""Fewer files than this are read in-process: starting workers would cost more."""


@dataclass(slots=True)
class LibraryPaths:
    """Where the catalog, stored songs and reports go."""

    catalog: Path
    scores: Path
    reports: Path
    annotations: Path | None = None
    """Defaults to an ``annotations`` folder next to ``scores``."""

    @classmethod
    def from_settings(cls, paths: PathSettings) -> LibraryPaths:
        return cls(
            catalog=paths.data_dir / "catalog.sqlite",
            scores=paths.data_dir / "scores",
            reports=paths.artifacts_dir / "reports",
            annotations=paths.data_dir / "annotations",
        )

    def score_path(self, song_id: str) -> Path:
        return self.scores / f"{song_id}.json.gz"

    def annotation_path(self, song_id: str) -> Path:
        folder = self.annotations or self.scores.parent / "annotations"
        return folder / f"{song_id}.json"


@dataclass(slots=True)
class FileResult:
    """What reading one file produced (returned from worker processes)."""

    record: SongRecord
    score: Score | None = None


def song_id_for(relative_path: str) -> str:
    """Stable id of a file inside the library."""
    return hashlib.sha1(relative_path.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def discover(root: Path) -> list[Path]:
    """Tab and MIDI files under ``root`` (plus known-but-unsupported ones, to report them)."""
    wanted = SUPPORTED_SUFFIXES | set(KNOWN_UNSUPPORTED)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted)


# ---------------------------------------------------------------- one file


def process_file(path: Path, root: Path, config: BacklineConfig) -> FileResult:
    """Read and analyse one file. Never raises for bad input: problems become the status."""
    relative = path.relative_to(root).as_posix()
    stat = path.stat()
    record = SongRecord(
        id=song_id_for(relative),
        path=relative,
        size=stat.st_size,
        mtime=stat.st_mtime,
        file_hash=file_digest(path),
        format=path.suffix.lower().lstrip("."),
        status="error",
        imported_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    stats = ReadStats()
    try:
        score = read_score(path, stats=stats)
    except UnsupportedFormatError as exc:
        record.status, record.reason = "unsupported", str(exc)
        return FileResult(record)
    except ScoreReadError as exc:
        record.reason = str(exc)
        return FileResult(record)
    except Exception as exc:
        log.exception("unexpected error while reading %s", relative)
        record.reason = f"{type(exc).__name__}: {exc}"
        return FileResult(record)

    record.read_stats = dict(stats)
    record.n_bars = score.n_bars
    record.tempo = score.tempos[0].bpm
    record.meters = ", ".join(sorted({str(ts) for ts in score.time_signatures}))
    hints = config.ingest.name_hints
    decisions = [classify_track(track, hints) for track in score.tracks]
    for idx, (track, decision) in enumerate(zip(score.tracks, decisions, strict=True)):
        pitches = [n.pitch for n in track.notes]
        record.tracks.append(
            TrackRecord(
                idx=idx,
                name=track.name,
                role=decision.role.value,
                confidence=decision.confidence,
                ambiguous=decision.ambiguous,
                reasons=list(decision.reasons),
                program=track.program,
                instrument=track.instrument,
                tuning=list(track.tuning) if track.tuning else None,
                n_notes=len(track.notes),
                pitch_min=min(pitches) if pitches else None,
                pitch_max=max(pitches) if pitches else None,
                stored_index=None,
            )
        )

    metadata = resolve_metadata(
        score,
        path,
        root,
        config.ingest.filename_patterns,
        folder_is_artist=config.ingest.folder_is_artist,
    )
    record.title, record.artist = metadata.title, metadata.artist
    record.title_source, record.artist_source = metadata.title_source, metadata.artist_source
    record.name_key = name_key(metadata)

    normalised, reasons = _normalise(score, decisions, record, config)
    if reasons:
        record.status, record.reason = "filtered", "; ".join(reasons)
        return FileResult(record)
    assert normalised is not None
    record.status = "ok"
    record.content_key = content_key(normalised)
    record.drum_stats = _drum_stats(normalised, config)
    return FileResult(record, normalised)


def _normalise(
    score: Score,
    decisions: list[RoleDecision],
    record: SongRecord,
    config: BacklineConfig,
) -> tuple[Score | None, list[str]]:
    ingest = config.ingest
    by_role: dict[TrackRole, list[int]] = {role: [] for role in TrackRole}
    for idx, decision in enumerate(decisions):
        by_role[decision.role].append(idx)
    tracks = score.tracks

    guitars = [
        i for i in by_role[TrackRole.GUITAR] if len(tracks[i].notes) >= ingest.min_notes_guitar
    ]
    basses = sorted(
        (i for i in by_role[TrackRole.BASS] if len(tracks[i].notes) >= ingest.min_notes_bass),
        key=lambda i: -len(tracks[i].notes),
    )
    drums = [i for i in by_role[TrackRole.DRUMS] if tracks[i].notes]
    drum_notes = sum(len(tracks[i].notes) for i in drums)

    reasons = []
    if not guitars:
        reasons.append("no guitar track")
    if not basses:
        reasons.append("no bass track")
    if drum_notes < ingest.min_notes_drums:
        reasons.append("no drum track")
    if reasons:
        return None, reasons
    bass_coverage = _coverage(score, [basses[0]], guitars)
    drum_coverage = _coverage(score, drums, guitars)
    record.read_stats["bass_coverage_pct"] = round(100 * bass_coverage)
    record.read_stats["drum_coverage_pct"] = round(100 * drum_coverage)
    if bass_coverage < ingest.min_coverage:
        reasons.append("bass covers too few bars")
    if drum_coverage < ingest.min_coverage:
        reasons.append("drums cover too few bars")
    if reasons:
        return None, reasons

    ranked = [p.index for p in rank_guitars(score, guitars, ingest.name_hints)]
    if len(basses) > 1:
        record.read_stats["extra_bass_tracks_ignored"] = len(basses) - 1
    if len(drums) > 1:
        record.read_stats["drum_tracks_merged"] = len(drums)

    kept: list[Track] = []
    for idx in ranked:
        source = tracks[idx]
        kept.append(
            Track(
                source.name,
                TrackRole.GUITAR,
                list(source.notes),
                source.program,
                False,
                source.tuning,
                source.instrument,
            )
        )
    bass = tracks[basses[0]]
    kept.append(
        Track(
            bass.name,
            TrackRole.BASS,
            list(bass.notes),
            bass.program,
            False,
            bass.tuning,
            bass.instrument,
        )
    )
    names: dict[int, str] = {}
    merged = []
    for idx in drums:
        merged.extend(tracks[idx].notes)
        for pitch, label in tracks[idx].percussion_names.items():
            names.setdefault(pitch, label)
    kept.append(
        Track(
            "Drums",
            TrackRole.DRUMS,
            sorted(merged),
            0,
            True,
            None,
            "drumKit",
            names,
        )
    )
    stored = {idx: pos for pos, idx in enumerate([*ranked, basses[0]])}
    stored.update({idx: len(kept) - 1 for idx in drums})
    for track_record in record.tracks:
        track_record.stored_index = stored.get(track_record.idx)
    record.n_guitars = len(ranked)

    normalised = Score(
        n_bars=score.n_bars,
        time_signatures=list(score.time_signatures),
        tempos=list(score.tempos),
        tracks=kept,
        tpq=score.tpq,
        title=record.title,
        artist=record.artist,
    )
    return normalised, []


def _bars_with_notes(score: Score, tracks: list[int]) -> set[int]:
    end = score.end
    return {score.bar_of(n.start) for i in tracks for n in score.tracks[i].notes if n.start < end}


def _coverage(score: Score, tracks: list[int], guitars: list[int]) -> float:
    """Share of the bars where a guitar plays that also have notes in ``tracks``.

    Tabs often transcribe the bass or drums only for an intro; such songs teach little.
    """
    reference = _bars_with_notes(score, guitars) or set(range(score.n_bars))
    return len(_bars_with_notes(score, tracks) & reference) / len(reference)


def _drum_stats(score: Score, config: BacklineConfig) -> dict[str, Any]:
    drums = score.tracks[-1]
    unmapped = Counter(n.pitch for n in drums.notes if config.drums.class_of(n.pitch) is None)
    return {
        "notes": len(drums.notes),
        "unmapped": {str(pitch): count for pitch, count in sorted(unmapped.items())},
        "names": {
            str(p): drums.percussion_names[p] for p in unmapped if p in drums.percussion_names
        },
    }


# ----------------------------------------------------------------- library


@dataclass(slots=True)
class IngestReport:
    """Summary of an import, also written as JSON next to the other reports."""

    source: str
    files: int
    read: int
    reused: int
    removed: int
    statuses: dict[str, int]
    reasons: dict[str, int]
    splits: dict[str, dict[str, int]]
    read_stats: dict[str, int]
    drums: dict[str, Any]
    ambiguous_tracks: list[dict[str, Any]] = field(default_factory=list)
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    report_path: str = ""

    def lines(self) -> list[str]:
        """Human-readable summary."""
        out = [
            f"library      {self.source}",
            f"files        {self.files} (read {self.read}, unchanged {self.reused}, "
            f"removed {self.removed})",
            "status       " + ", ".join(f"{k}={v}" for k, v in sorted(self.statuses.items())),
        ]
        if self.reasons:
            out.append(
                "skipped      " + ", ".join(f"{k}: {v}" for k, v in sorted(self.reasons.items()))
            )
        for split, counts in sorted(self.splits.items()):
            out.append(f"split {split:<6} {counts['songs']} songs in {counts['groups']} groups")
        if self.read_stats:
            out.append(
                "read notes   " + ", ".join(f"{k}={v}" for k, v in sorted(self.read_stats.items()))
            )
        share = self.drums.get("unmapped_share", 0.0)
        out.append(f"drums        {self.drums.get('notes', 0)} notes, unmapped {share:.1%}")
        if self.drums.get("warning"):
            out.append(f"WARNING      {self.drums['warning']}")
        if self.ambiguous_tracks:
            out.append(f"ambiguous    {len(self.ambiguous_tracks)} tracks (see report)")
        if self.duplicate_groups:
            out.append(f"duplicates   {len(self.duplicate_groups)} songs with several versions")
        if self.report_path:
            out.append(f"report       {self.report_path}")
        return out


def run_ingest(
    source: Path,
    config: BacklineConfig,
    library: LibraryPaths,
    *,
    workers: int | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> IngestReport:
    """Import (or refresh) the library under ``source``."""
    root = source.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"tab library folder not found: {root}")
    files = discover(root)
    with Catalog(library.catalog) as catalog:
        previous = {s.id: s for s in catalog.songs()}
        reused: dict[str, SongRecord] = {}
        pending: list[Path] = []
        for path in files:
            song_id = song_id_for(path.relative_to(root).as_posix())
            old = previous.get(song_id)
            stat = path.stat()
            unchanged = old is not None and old.size == stat.st_size and old.mtime == stat.st_mtime
            stored_ok = old is not None and (
                old.status not in ("ok", "duplicate") or library.score_path(song_id).is_file()
            )
            if not force and unchanged and stored_ok and old is not None:
                reused[song_id] = old
            else:
                pending.append(path)

        results = _read_all(pending, root, config, workers, progress)
        for result in results:
            if result.score is not None:
                save_score(result.score, library.score_path(result.record.id))
        records = {**reused, **{r.record.id: r.record for r in results}}

        removed = [song_id for song_id in previous if song_id not in records]
        for song_id in removed:
            library.score_path(song_id).unlink(missing_ok=True)
            library.annotation_path(song_id).unlink(missing_ok=True)

        _deduplicate_and_split(records, config)
        with catalog.transaction():
            catalog.remove(removed)
            catalog.upsert(records.values())
            catalog.set_meta("source", str(root))
            catalog.set_meta("imported_at", datetime.now(UTC).isoformat(timespec="seconds"))

    report = _report(root, records, len(files), len(pending), len(reused), len(removed), config)
    library.reports.mkdir(parents=True, exist_ok=True)
    path = library.reports / "ingest_latest.json"
    report.report_path = str(path)
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def _read_all(
    paths: list[Path],
    root: Path,
    config: BacklineConfig,
    workers: int | None,
    progress: ProgressCallback | None,
) -> list[FileResult]:
    count = workers if workers is not None else config.ingest.workers
    if count == 0:
        count = min(8, os.cpu_count() or 1)
    results: list[FileResult] = []
    if count <= 1 or len(paths) < PARALLEL_THRESHOLD or not _spawn_safe():
        for done, path in enumerate(paths, start=1):
            results.append(process_file(path, root, config))
            if progress:
                progress(done, len(paths))
        return results
    with ProcessPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(process_file, path, root, config) for path in paths]
        for done, future in enumerate(futures, start=1):
            results.append(future.result())
            if progress:
                progress(done, len(paths))
    return results


def _spawn_safe() -> bool:
    """Whether worker processes can be started.

    Windows starts workers by re-importing the main module; code run from stdin or an
    interactive prompt has none, and the pool would hang, so read in-process instead.
    """
    import __main__

    return getattr(__main__, "__file__", None) is not None


def _completeness(record: SongRecord) -> tuple[int, int, int, str]:
    """Sort key: the most complete version of a song comes first."""
    kept_notes = sum(t.n_notes for t in record.tracks if t.stored_index is not None)
    return (-record.n_guitars, -record.n_bars, -kept_notes, record.path)


def _deduplicate_and_split(records: dict[str, SongRecord], config: BacklineConfig) -> None:
    candidates = [r for r in records.values() if r.status in ("ok", "duplicate")]
    for record in records.values():
        if record.status not in ("ok", "duplicate"):
            record.group_id, record.split, record.canonical = None, None, False
    groups = group_songs(
        [SongKeys(r.id, r.name_key, r.content_key or r.file_hash) for r in candidates]
    )
    members: dict[str, list[SongRecord]] = {}
    for record in candidates:
        record.group_id = groups[record.id]
        members.setdefault(record.group_id, []).append(record)
    ingest = config.ingest
    for group_id, versions in members.items():
        versions.sort(key=_completeness)
        split = assign_split(
            group_id,
            config.seed,
            val_fraction=ingest.val_fraction,
            test_fraction=ingest.test_fraction,
        )
        for position, record in enumerate(versions):
            record.split = split
            record.canonical = position == 0
            record.status = "ok" if position == 0 or ingest.keep_duplicates else "duplicate"
            record.reason = "" if record.status == "ok" else f"version of {versions[0].path}"


def _report(
    root: Path,
    records: dict[str, SongRecord],
    files: int,
    read: int,
    reused: int,
    removed: int,
    config: BacklineConfig,
) -> IngestReport:
    statuses = Counter(r.status for r in records.values())
    reasons: Counter[str] = Counter()
    for record in records.values():
        if record.status in ("filtered", "unsupported", "error"):
            for reason in record.reason.split("; "):
                key = reason if record.status == "filtered" else record.status
                reasons[key] += 1
    splits: dict[str, dict[str, int]] = {}
    for record in records.values():
        if record.status == "ok" and record.split:
            entry = splits.setdefault(record.split, {"songs": 0, "groups": 0})
            entry["songs"] += 1
    for split in splits:
        splits[split]["groups"] = len(
            {r.group_id for r in records.values() if r.status == "ok" and r.split == split}
        )

    read_stats: Counter[str] = Counter()
    drum_notes = 0
    unmapped: Counter[str] = Counter()
    names: dict[str, str] = {}
    for record in records.values():
        if record.status != "ok":
            continue
        read_stats.update(record.read_stats)
        drum_notes += int(record.drum_stats.get("notes", 0))
        unmapped.update({k: int(v) for k, v in record.drum_stats.get("unmapped", {}).items()})
        names.update(record.drum_stats.get("names", {}))
    share = sum(unmapped.values()) / drum_notes if drum_notes else 0.0
    threshold = config.drums.other_percussion_warn_ratio
    drums: dict[str, Any] = {
        "notes": drum_notes,
        "unmapped_share": round(share, 4),
        "unmapped": [
            {"note": int(note), "name": names.get(note, ""), "count": count}
            for note, count in unmapped.most_common()
        ],
    }
    if share > threshold:
        drums["warning"] = (
            f"{share:.1%} of drum notes are not mapped to a class (threshold {threshold:.0%}); "
            "consider extending drums.input_map"
        )

    ambiguous = [
        {
            "path": record.path,
            "track": track.name,
            "role": track.role,
            "kept": track.stored_index is not None,
            "reasons": track.reasons,
        }
        for record in records.values()
        for track in record.tracks
        if track.ambiguous and track.n_notes
    ]
    groups: dict[str, list[SongRecord]] = {}
    for record in records.values():
        if record.group_id:
            groups.setdefault(record.group_id, []).append(record)
    duplicate_groups = [
        {
            "group": group_id,
            "kept": next(r.path for r in versions if r.canonical),
            "versions": sorted(r.path for r in versions),
        }
        for group_id, versions in sorted(groups.items())
        if len(versions) > 1
    ]
    return IngestReport(
        source=str(root),
        files=files,
        read=read,
        reused=reused,
        removed=removed,
        statuses=dict(statuses),
        reasons=dict(reasons),
        splits=splits,
        read_stats=dict(read_stats),
        drums=drums,
        ambiguous_tracks=ambiguous,
        duplicate_groups=duplicate_groups,
    )


def load_song(library: LibraryPaths, song_id: str) -> Score:
    """Stored, normalised score of an imported song."""
    return load_score(library.score_path(song_id))
