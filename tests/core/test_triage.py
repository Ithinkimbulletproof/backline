# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Triage of imported tabs: what to fix by hand and what to set aside."""

from __future__ import annotations

from pathlib import Path

import pytest

from backline.config import BacklineConfig
from backline.ingest.catalog import Catalog, SongRecord, TrackRecord
from backline.ingest.pipeline import LibraryPaths, run_ingest
from backline.ingest.triage import Category, Problem, count, triage
from tests.library import build_library


def _track(
    idx: int, role: str, *, notes: int = 100, low: int = 40, high: int = 76, **kw: object
) -> TrackRecord:
    return TrackRecord(
        idx=idx,
        name=str(kw.get("name", f"Track {idx + 1}")),
        role=role,
        confidence=0.9,
        ambiguous=bool(kw.get("ambiguous", False)),
        reasons=list(kw.get("reasons", [])),  # type: ignore[call-overload]
        program=0,
        instrument="",
        tuning=None,
        n_notes=notes,
        pitch_min=low,
        pitch_max=high,
        stored_index=kw.get("stored"),  # type: ignore[arg-type]
    )


def _record(
    status: str, reason: str = "", tracks: list[TrackRecord] | None = None, **kw: object
) -> SongRecord:
    return SongRecord(
        id="x",
        path="x.gp",
        size=1,
        mtime=0.0,
        file_hash="h",
        format=str(kw.pop("format", "gp")),
        status=status,
        reason=reason,
        artist=str(kw.pop("artist", "Band")),
        artist_source=str(kw.pop("artist_source", "tags")),
        tracks=tracks or [],
        **kw,  # type: ignore[arg-type]
    )


def test_synthetic_library(tmp_path: Path, tiny_config: BacklineConfig) -> None:
    paths = build_library(tmp_path / "tabs", tiny_config)
    store = LibraryPaths(tmp_path / "catalog.sqlite", tmp_path / "scores", tmp_path / "reports")
    report = run_ingest(tmp_path / "tabs", tiny_config, store, workers=1)
    with Catalog(store.catalog) as catalog:
        found = {r.path: triage(r, tiny_config) for r in catalog.songs()}

    assert found[paths["gpx"]].category is Category.FIXABLE
    assert found[paths["gpx"]].problems[0].code == "old_format"
    assert found[paths["broken"]].category is Category.FIXABLE
    assert found[paths["no_bass"]].category is Category.REJECTED
    assert found[paths["no_bass"]].problems[0].code == "no_bass"
    intro = found[paths["bass_intro"]]
    assert intro.category is Category.REJECTED
    assert intro.problems[0].code == "bass_mostly_missing"
    assert found[paths["cowbell"]].category is Category.REVIEW
    assert found[paths["cowbell"]].problems[0].code == "unmapped_percussion"
    mislabelled = found[paths["ambiguous"]]
    assert mislabelled.category is Category.REVIEW
    assert "'Track 3' was taken for the bass" in mislabelled.summary()
    assert found[paths["a_copy"]].category is Category.DUPLICATE
    assert found[paths["a"]].category is Category.OK
    assert report.triage == count(found.values())
    assert list(report.triage) == ["fixable", "rejected", "review", "ok", "duplicate"]
    assert any(line.startswith("triage") for line in report.lines())


def test_unrecognised_bass_is_fixable(tiny_config: BacklineConfig) -> None:
    tracks = [
        _track(0, "guitar", name="Rhythm"),
        _track(1, "guitar", name="Guitar 2", low=28, high=55, reasons=["program 29 -> guitar"]),
        _track(2, "drums", low=36, high=57),
    ]
    result = triage(_record("filtered", "no bass track", tracks), tiny_config)
    assert result.category is Category.FIXABLE
    problem = result.problems[0]
    assert (problem.code, problem.track, problem.detail) == ("bass_not_recognised", 1, "guitar")
    assert "track 2 'Guitar 2' looks like the bass" in problem.describe()
    assert "name it 'Bass'" in problem.describe()


def test_the_only_guitar_is_not_offered_as_the_bass(tiny_config: BacklineConfig) -> None:
    tracks = [_track(0, "guitar", low=33, high=58), _track(1, "drums")]
    result = triage(_record("filtered", "no bass track", tracks), tiny_config)
    assert result.category is Category.REJECTED
    assert result.problems[0].code == "no_bass"


def test_thin_bass_track_is_named(tiny_config: BacklineConfig) -> None:
    tracks = [_track(0, "guitar"), _track(1, "bass", notes=5, low=28, high=40), _track(2, "drums")]
    result = triage(_record("filtered", "no bass track", tracks), tiny_config)
    assert result.problems == (Problem("no_bass", value=5.0),)
    assert result.problems[0].describe() == "no bass part (the track has only 5 notes)"


def test_drums_voted_by_name_are_offered(tiny_config: BacklineConfig) -> None:
    tracks = [
        _track(0, "guitar"),
        _track(1, "bass", low=28, high=50),
        _track(2, "other", name="Drums", reasons=["program 0 -> guitar", "name 'Drums' -> drums"]),
    ]
    result = triage(_record("filtered", "no drum track", tracks), tiny_config)
    assert result.category is Category.FIXABLE
    assert result.problems[0].code == "drums_not_recognised"


@pytest.mark.parametrize(("pct", "code", "category"), [
    (45, "bass_coverage_low", Category.FIXABLE),
    (20, "bass_mostly_missing", Category.REJECTED),
])  # fmt: skip
def test_coverage_decides_between_fixing_and_rejecting(
    tiny_config: BacklineConfig, pct: int, code: str, category: Category
) -> None:
    record = _record("filtered", "bass covers too few bars", read_stats={"bass_coverage_pct": pct})
    result = triage(record, tiny_config)
    assert result.category is category
    assert result.problems[0].code == code
    if code == "bass_coverage_low":
        assert "45% of the guitar's bars (60% needed)" in result.problems[0].describe()


def test_one_hopeless_problem_rejects_the_file(tiny_config: BacklineConfig) -> None:
    record = _record(
        "filtered",
        "bass covers too few bars; drums cover too few bars",
        read_stats={"bass_coverage_pct": 50, "drum_coverage_pct": 10},
    )
    result = triage(record, tiny_config)
    assert [p.code for p in result.problems] == ["bass_coverage_low", "drums_mostly_missing"]
    assert result.category is Category.REJECTED


def test_review_flags(tiny_config: BacklineConfig) -> None:
    tracks = [_track(0, "guitar", stored=0), _track(1, "bass", ambiguous=True, stored=1)]
    record = _record(
        "ok",
        tracks=tracks,
        artist="",
        artist_source="none",
        read_stats={"extra_bass_tracks_ignored": 1},
    )
    codes = [p.code for p in triage(record, tiny_config).problems]
    assert codes == ["ambiguous_role", "extra_bass_ignored", "no_artist"]
    assert triage(_record("ok"), tiny_config).category is Category.OK


def test_problems_round_trip_as_dicts() -> None:
    problem = Problem("bass_coverage_low", value=0.45, limit=0.6)
    assert Problem(**problem.to_dict()) == problem
