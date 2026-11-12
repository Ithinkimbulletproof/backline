# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end import of a synthetic tab library."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backline.config import BacklineConfig
from backline.ingest.catalog import Catalog, SongRecord
from backline.ingest.pipeline import IngestReport, LibraryPaths, load_song, run_ingest
from backline.score import TrackRole
from tests.library import build_library

Library = tuple[Path, dict[str, str], LibraryPaths]


@pytest.fixture
def library(tmp_path: Path, tiny_config: BacklineConfig) -> Library:
    root = tmp_path / "tabs"
    paths = build_library(root, tiny_config)
    store = LibraryPaths(
        catalog=tmp_path / "data" / "catalog.sqlite",
        scores=tmp_path / "data" / "scores",
        reports=tmp_path / "artifacts" / "reports",
    )
    return root, paths, store


def _records(store: LibraryPaths) -> dict[str, SongRecord]:
    with Catalog(store.catalog) as catalog:
        return {s.path: s for s in catalog.songs()}


def _ingest(library: Library, config: BacklineConfig, **kwargs: object) -> IngestReport:
    root, _, store = library
    return run_ingest(root, config, store, workers=1, **kwargs)  # type: ignore[arg-type]


def test_statuses_and_reasons(library: Library, tiny_config: BacklineConfig) -> None:
    report = _ingest(library, tiny_config)
    _, paths, store = library
    records = _records(store)
    assert report.files == len(records) == 18
    assert records[paths["no_bass"]].status == "filtered"
    assert records[paths["no_bass"]].reason == "no bass track"
    assert records[paths["gpx"]].status == "unsupported"
    assert records[paths["broken"]].status == "error"
    assert "notes.txt" not in records
    assert report.statuses == {
        "ok": 10,
        "duplicate": 4,
        "filtered": 2,
        "unsupported": 1,
        "error": 1,
    }
    assert report.reasons == {
        "no bass track": 1,
        "bass covers too few bars": 1,
        "unsupported": 1,
        "error": 1,
    }
    assert records[paths["bass_intro"]].reason == "bass covers too few bars"
    assert records[paths["a"]].read_stats["bass_coverage_pct"] >= 60


def test_versions_of_a_song_are_grouped(library: Library, tiny_config: BacklineConfig) -> None:
    report = _ingest(library, tiny_config)
    _, paths, store = library
    records = _records(store)
    for first, second in (
        ("a", "a_copy"),
        ("b", "b_live"),
        ("c", "c_other"),
        ("cyrillic", "latin"),
    ):
        one, two = records[paths[first]], records[paths[second]]
        assert one.group_id == two.group_id, (first, second)
        assert one.split == two.split
        assert one.canonical != two.canonical
        assert {one.status, two.status} == {"ok", "duplicate"}
    assert len(report.duplicate_groups) == 4
    assert records[paths["b_live"]].title == "Low Tide (Live)"
    assert records[paths["b"]].artist == "Grey Harbor"
    assert records[paths["b"]].artist_source == "tags"


def test_split_keeps_groups_together(library: Library, tiny_config: BacklineConfig) -> None:
    _ingest(library, tiny_config)
    _, _, store = library
    by_group: dict[str, set[str]] = {}
    for record in _records(store).values():
        if record.group_id:
            by_group.setdefault(record.group_id, set()).add(str(record.split))
    assert all(len(splits) == 1 for splits in by_group.values())


def test_stored_songs_are_normalised(library: Library, tiny_config: BacklineConfig) -> None:
    _ingest(library, tiny_config)
    _, paths, store = library
    record = _records(store)[paths["lead_first"]]
    score = load_song(store, record.id)
    roles = [t.role for t in score.tracks]
    assert roles == [TrackRole.GUITAR, TrackRole.GUITAR, TrackRole.BASS, TrackRole.DRUMS]
    assert score.tracks[0].name == paths["rhythm_name"]
    assert score.tracks[-1].is_percussion
    assert record.n_guitars == 2
    stored = {t.name: t.stored_index for t in record.tracks}
    assert stored[paths["rhythm_name"]] == 0


def test_ambiguous_tracks_and_unmapped_drums_are_reported(
    library: Library, tiny_config: BacklineConfig
) -> None:
    report = _ingest(library, tiny_config)
    _, paths, store = library
    ambiguous = [a for a in report.ambiguous_tracks if a["path"] == paths["ambiguous"]]
    assert [(a["track"], a["role"], a["kept"]) for a in ambiguous] == [("Track 3", "bass", True)]
    cowbell = next(u for u in report.drums["unmapped"] if u["note"] == 56)
    assert cowbell["count"] > 0
    assert report.drums["unmapped_share"] > 0
    assert records_ok(store)
    assert any("unmapped" in line for line in report.lines())


def records_ok(store: LibraryPaths) -> bool:
    records = _records(store)
    return all(
        store.score_path(r.id).is_file() == (r.status in ("ok", "duplicate"))
        for r in records.values()
    )


def test_reimport_reuses_unchanged_files_and_drops_removed_ones(
    library: Library, tiny_config: BacklineConfig
) -> None:
    root, paths, store = library
    first = _ingest(library, tiny_config)
    second = _ingest(library, tiny_config)
    assert (second.read, second.reused) == (0, first.files)

    changed = root / paths["a"]
    stat = changed.stat()
    os.utime(changed, (stat.st_atime, stat.st_mtime + 10))
    removed_id = _records(store)[paths["c"]].id
    (root / paths["c"]).unlink()
    third = _ingest(library, tiny_config)
    assert (third.read, third.removed) == (1, 1)
    assert paths["c"] not in _records(store)
    assert not store.score_path(removed_id).exists()
    assert _records(store)[paths["c_other"]].status == "ok"


def test_keep_duplicates(library: Library, tiny_config: BacklineConfig) -> None:
    config = tiny_config.model_copy(
        update={"ingest": tiny_config.ingest.model_copy(update={"keep_duplicates": True})}
    )
    report = _ingest(library, config)
    assert report.statuses.get("duplicate", 0) == 0
    assert report.statuses["ok"] == 14


def test_report_is_written(library: Library, tiny_config: BacklineConfig) -> None:
    report = _ingest(library, tiny_config)
    path = Path(report.report_path)
    assert path.is_file()
    assert '"duplicate_groups"' in path.read_text(encoding="utf-8")


def test_parallel_reading_matches_sequential(
    library: Library, tiny_config: BacklineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import backline.ingest.pipeline as pipeline

    root, _, store = library
    sequential = run_ingest(root, tiny_config, store, workers=1)
    monkeypatch.setattr(pipeline, "PARALLEL_THRESHOLD", 0)
    parallel = run_ingest(root, tiny_config, store, workers=2, force=True)
    assert parallel.read == sequential.files
    assert parallel.statuses == sequential.statuses
    assert parallel.duplicate_groups == sequential.duplicate_groups


def test_missing_library_folder(tmp_path: Path, tiny_config: BacklineConfig) -> None:
    store = LibraryPaths(tmp_path / "c.sqlite", tmp_path / "s", tmp_path / "r")
    with pytest.raises(FileNotFoundError):
        run_ingest(tmp_path / "nope", tiny_config, store)
