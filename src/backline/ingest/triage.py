# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Sort imported tabs by what it takes to use them, with a concrete hint for each file.

The import decides only whether a file goes into the dataset. Triage reads the same
catalog records and tells apart what is worth fixing by hand from what is not:

- ``ok``: in the dataset as it is;
- ``review``: in the dataset, but something deserves a look (an uncertain track role,
  dropped percussion, a missing artist);
- ``fixable``: left out, and a small edit would let it in (a bass track the vote did not
  recognise, a part a little short of the coverage threshold, an old file format);
- ``rejected``: nothing to learn from (no bass or no drums at all, parts mostly missing);
- ``duplicate``: another version of a song that is already in the dataset.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from backline.config import BacklineConfig, IngestConfig
from backline.ingest.catalog import SongRecord, TrackRecord

GUITAR_LOWEST = 40
"""E2, the lowest note of a standard-tuned guitar."""
BASS_REGISTER_TOP = 62
"""Bass lines rarely go above D4; guitar parts nearly always do."""


class Category(StrEnum):
    """What it takes to use a file."""

    OK = "ok"
    REVIEW = "review"
    FIXABLE = "fixable"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


CATEGORY_ORDER = (
    Category.FIXABLE,
    Category.REJECTED,
    Category.REVIEW,
    Category.OK,
    Category.DUPLICATE,
)
"""The order reports list categories in: work to do first."""

FIXABLE_CODES = frozenset(
    {
        "old_format",
        "read_error",
        "bass_not_recognised",
        "drums_not_recognised",
        "guitar_not_recognised",
        "bass_coverage_low",
        "drums_coverage_low",
    }
)
REJECTED_CODES = frozenset(
    {
        "unsupported_format",
        "no_bass",
        "no_drums",
        "no_guitar",
        "bass_mostly_missing",
        "drums_mostly_missing",
    }
)

_ROLE_WORDS = {"guitar": "a guitar", "bass": "the bass", "drums": "drums", "other": "other"}


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing found in a file. ``track`` is 0-based; texts number tracks from 1."""

    code: str
    track: int | None = None
    track_name: str = ""
    value: float | None = None
    """A share (coverage, dropped percussion) or a count, depending on the code."""
    limit: float | None = None
    """The threshold ``value`` was compared with, when there is one."""
    detail: str = ""

    def describe(self) -> str:
        """The problem and what to do about it, in one English sentence."""
        track = f"track {(self.track or 0) + 1} {self.track_name!r}"
        share = f"{self.value:.0%}" if self.value is not None else "?"
        need = f"{self.limit:.0%}" if self.limit is not None else "?"
        thin = f" (the track has only {int(self.value)} notes)" if self.value else ""
        texts = {
            "old_format": "Guitar Pro 6 file: open it in Guitar Pro 7 or 8 and save it as .gp",
            "read_error": f"cannot be read ({self.detail}); re-saving it in Guitar Pro may help",
            "bass_not_recognised": f"{track} looks like the bass but was taken for "
            f"{_ROLE_WORDS.get(self.detail, self.detail)}: if it is the bass, name it 'Bass'",
            "drums_not_recognised": f"{track} looks like drums but was taken for "
            f"{_ROLE_WORDS.get(self.detail, self.detail)}: if it is the drums, name it 'Drums'",
            "guitar_not_recognised": f"{track} looks like a guitar but was taken for "
            f"{_ROLE_WORDS.get(self.detail, self.detail)}: if it is a guitar, name it 'Guitar'",
            "bass_coverage_low": f"the bass plays in {share} of the guitar's bars ({need} needed): "
            "fill in the missing bars",
            "drums_coverage_low": f"the drums play in {share} of the guitar's bars "
            f"({need} needed): fill in the missing bars",
            "unsupported_format": f"unsupported format: {self.detail}",
            "no_bass": "no bass part" + thin,
            "no_drums": "no drum part" + thin,
            "no_guitar": "no guitar part" + thin,
            "bass_mostly_missing": f"the bass plays in only {share} of the guitar's bars",
            "drums_mostly_missing": f"the drums play in only {share} of the guitar's bars",
            "ambiguous_role": f"{track} was taken for {_ROLE_WORDS.get(self.detail, self.detail)}"
            " but the clues disagree: check it",
            "unmapped_percussion": f"{share} of the drum notes are other percussion and are not "
            f"used ({self.detail})",
            "no_artist": "no artist in the tab or the file name: name the file 'Artist - Title'",
            "extra_bass_ignored": f"{int(self.value or 0)} more bass track(s) ignored, the fullest "
            "one is used: check it is the right one",
            "version_of": f"another version of {self.detail}",
        }
        return texts.get(self.code, self.code)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Triage:
    """The category of a file and everything that put it there."""

    category: Category
    problems: tuple[Problem, ...] = ()

    def summary(self) -> str:
        return "; ".join(p.describe() for p in self.problems)


def triage(record: SongRecord, config: BacklineConfig) -> Triage:
    """Sort one catalog record."""
    if record.status == "duplicate":
        kept = record.reason.removeprefix("version of ").strip()
        return Triage(Category.DUPLICATE, (Problem("version_of", detail=kept),))
    if record.status == "unsupported":
        code = "old_format" if record.format == "gpx" else "unsupported_format"
        return _sorted((Problem(code, detail=record.reason),))
    if record.status == "error":
        return _sorted((Problem("read_error", detail=record.reason),))
    if record.status == "filtered":
        return _sorted(tuple(_filtered(record, config)))
    review = tuple(_review(record, config))
    return Triage(Category.REVIEW if review else Category.OK, review)


def count(triages: Iterable[Triage]) -> dict[str, int]:
    """Files per category, in report order (categories without files included)."""
    totals = dict.fromkeys((c.value for c in CATEGORY_ORDER), 0)
    for item in triages:
        totals[item.category.value] += 1
    return totals


def _sorted(problems: tuple[Problem, ...]) -> Triage:
    """Rejected if anything rules the file out, fixable if every problem can be fixed."""
    if any(p.code in REJECTED_CODES for p in problems):
        return Triage(Category.REJECTED, problems)
    return Triage(Category.FIXABLE, problems)


def _filtered(record: SongRecord, config: BacklineConfig) -> list[Problem]:
    ingest = config.ingest
    reasons = set(record.reason.split("; "))
    problems: list[Problem] = []
    for role, reason in (
        ("guitar", "no guitar track"),
        ("bass", "no bass track"),
        ("drums", "no drum track"),
    ):
        if reason in reasons:
            problems.extend(_missing(record, role, ingest))
    for part, reason in (
        ("bass", "bass covers too few bars"),
        ("drums", "drums cover too few bars"),
    ):
        if reason in reasons:
            share = record.read_stats.get(f"{part}_coverage_pct", 0) / 100
            low = share >= ingest.fixable_coverage
            code = f"{part}_coverage_low" if low else f"{part}_mostly_missing"
            problems.append(Problem(code, value=share, limit=ingest.min_coverage))
    return problems


def _missing(record: SongRecord, role: str, ingest: IngestConfig) -> list[Problem]:
    """A part the import did not find: a track that may be it, or nothing to work with."""
    min_notes = {
        "guitar": ingest.min_notes_guitar,
        "bass": ingest.min_notes_bass,
        "drums": ingest.min_notes_drums,
    }[role]
    guitars = [
        t for t in record.tracks if t.role == "guitar" and t.n_notes >= ingest.min_notes_guitar
    ]
    candidates = [
        t
        for t in record.tracks
        if t.role != role
        and (role == "drums" or t.role != "drums")
        and t.n_notes >= min_notes
        and _looks_like(t, role)
        # Renaming the only guitar would just move the problem.
        and not (t.role == "guitar" and len(guitars) <= 1)
    ]
    if candidates:
        best = max(candidates, key=lambda t: t.n_notes)
        return [Problem(f"{role}_not_recognised", best.idx, best.name, detail=best.role)]
    thin = [t.n_notes for t in record.tracks if t.role == role and t.n_notes]
    code = {"guitar": "no_guitar", "bass": "no_bass", "drums": "no_drums"}[role]
    # value: notes of the nearly empty track the import found, if any.
    return [Problem(code, value=float(max(thin)) if thin else None)]


def _looks_like(track: TrackRecord, role: str) -> bool:
    voted = any(reason.endswith(f"-> {role}") for reason in track.reasons)
    if role != "bass" or voted:
        return voted
    low, high = track.pitch_min, track.pitch_max
    return (
        low is not None and high is not None and low < GUITAR_LOWEST and high <= BASS_REGISTER_TOP
    )


def _review(record: SongRecord, config: BacklineConfig) -> list[Problem]:
    problems = [
        Problem("ambiguous_role", t.idx, t.name, detail=t.role)
        for t in record.tracks
        if t.ambiguous and t.n_notes and t.stored_index is not None
    ]
    extra = record.read_stats.get("extra_bass_tracks_ignored", 0)
    if extra:
        problems.append(Problem("extra_bass_ignored", value=float(extra)))
    notes = int(record.drum_stats.get("notes", 0))
    unmapped = record.drum_stats.get("unmapped", {})
    dropped = sum(int(v) for v in unmapped.values())
    share = dropped / notes if notes else 0.0
    if share > config.drums.other_percussion_warn_ratio:
        names = record.drum_stats.get("names", {})
        labels = ", ".join(names.get(k, f"note {k}") for k in sorted(unmapped, key=int))
        problems.append(Problem("unmapped_percussion", value=round(share, 4), detail=labels))
    if record.artist_source == "none" or not record.artist:
        problems.append(Problem("no_artist"))
    return problems


# ------------------------------------------------------------------ the whole library


@dataclass(frozen=True, slots=True)
class TriagedSong:
    """A catalog record and its triage."""

    record: SongRecord
    triage: Triage


def triage_library(records: Iterable[SongRecord], config: BacklineConfig) -> list[TriagedSong]:
    """Every record with its triage, work to do first, then by path."""
    order = {category: i for i, category in enumerate(CATEGORY_ORDER)}
    items = [TriagedSong(record, triage(record, config)) for record in records]
    return sorted(items, key=lambda item: (order[item.triage.category], item.record.path))


CSV_COLUMNS = ("category", "path", "artist", "title", "problems", "codes")


def write_csv(items: Sequence[TriagedSong], path: Path) -> None:
    """One row per file: its category, what is wrong and what to do (UTF-8, comma-separated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for item in items:
            record = item.record
            writer.writerow(
                (
                    item.triage.category.value,
                    record.path,
                    record.artist,
                    record.title,
                    item.triage.summary(),
                    " ".join(p.code for p in item.triage.problems),
                )
            )


class MoveError(ValueError):
    """Raised when files cannot be moved where asked."""


@dataclass(slots=True)
class MoveResult:
    """Files moved (or that would be, in a dry run) and files left where they are, with why."""

    moved: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def move_files(
    items: Sequence[TriagedSong], root: Path, target: Path, *, dry_run: bool = False
) -> MoveResult:
    """Move the files of ``items`` from the library ``root`` to ``target``, keeping subfolders.

    Nothing is deleted and nothing is overwritten: a file already at the destination is
    skipped. ``target`` must lie outside the library, or the next import would find the
    files again.
    """
    root, target = root.resolve(), target.resolve()
    if target == root or target.is_relative_to(root):
        raise MoveError(f"choose a folder outside the library ({root}), not {target}")
    result = MoveResult()
    for item in items:
        source = root / item.record.path
        destination = target / item.record.path
        if not source.is_file():
            result.skipped.append((source, "not found"))
            continue
        if destination.exists():
            result.skipped.append((source, f"{destination} already exists"))
            continue
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
        result.moved.append((source, destination))
    return result
