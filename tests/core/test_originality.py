# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Copy detection: invariance, planted plagiarism, common vocabulary, references, calibration."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backline.config import BacklineConfig
from backline.formats.guitarpro import write_gp5
from backline.formats.store import load_score
from backline.ingest.catalog import Catalog
from backline.ingest.pipeline import LibraryPaths, run_ingest
from backline.originality.build import build_originality_index, slice_score
from backline.originality.check import OriginalityChecker
from backline.originality.index import OriginalityIndex, ngram_hashes
from backline.originality.represent import bass_events, rescale
from backline.score import TPQ, Note, Score, Tempo, TimeSignature, Track, TrackRole
from backline.synth import generate_corpus
from tests.library import build_library

E = TPQ // 2
BAR = 4 * TPQ
SIGNATURE_FILL = [
    (0, 36),
    (240, 38),
    (360, 38),
    (600, 47),
    (720, 45),
    (840, 43),
    (1080, 49),
    (1200, 41),
    (1560, 52),
]


def _with_signature_fill(score: Score) -> Score:
    """Replace the drums by a distinctive, irregular pattern found in no other song."""
    result = copy.deepcopy(score)
    kit = next(t for t in result.tracks if t.is_percussion)
    kit.notes = sorted(
        Note(bar * BAR + offset + (bar % 3) * 120, pitch, 120)
        for bar in range(result.n_bars)
        for offset, pitch in SIGNATURE_FILL
    )
    return result


@pytest.fixture(scope="module")
def corpus_index(
    tmp_path_factory: pytest.TempPathFactory, tiny_config: BacklineConfig
) -> tuple[OriginalityIndex, LibraryPaths, list[str]]:
    root = tmp_path_factory.mktemp("originality")
    tabs = root / "tabs"
    tabs.mkdir()
    songs = generate_corpus(tiny_config, n_songs=40, seed=12)
    for index, song in enumerate(songs):
        score = _with_signature_fill(song.score) if index == 7 else song.score
        write_gp5(score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    library = LibraryPaths(root / "catalog.sqlite", root / "scores", root / "reports")
    run_ingest(tabs, tiny_config, library, workers=1)
    with Catalog(library.catalog) as catalog:
        ids = [s.id for s in sorted(catalog.songs("ok"), key=lambda s: s.path)]
    return build_originality_index(tiny_config, library, root / "index"), library, ids


def _disguise(score: Score, semitones: int, stretch: int, edits: tuple[int, ...] = ()) -> Score:
    result = copy.deepcopy(score)
    for track in result.tracks:
        notes = []
        for k, note in enumerate(sorted(track.notes)):
            pitch = note.pitch if track.is_percussion else note.pitch + semitones
            if track.role is TrackRole.BASS and k in edits:
                pitch += 2
            notes.append(
                replace(
                    note, start=note.start * stretch, duration=note.duration * stretch, pitch=pitch
                )
            )
        track.notes = notes
    return Score(result.n_bars * stretch, result.time_signatures, result.tempos, result.tracks)


def _generic() -> Score:
    bass = [Note(i * E, 28 + (5 if (i // 8) % 4 == 1 else 0), E) for i in range(64)]
    beat = [(0, 36), (0, 42), (480, 42), (960, 38), (960, 42), (1440, 42)]
    drums = [
        Note(b * BAR + o + h * 1920, p, 120) for b in range(8) for h in (0, 1) for o, p in beat
    ]
    return Score(
        8,
        [TimeSignature(0, 4, 4)],
        [Tempo(0, 120.0)],
        [
            Track("Bass", TrackRole.BASS, bass, 33),
            Track("Drums", TrackRole.DRUMS, drums, is_percussion=True),
        ],
    )


# --------------------------------------------------------------- representations


def test_bass_events_ignore_key_tempo_and_half_time() -> None:
    line = [
        Note(i * E, p, E if i % 3 else 2 * E)
        for i, p in enumerate([28, 31, 33, 35, 33, 31, 28, 26, 28, 40])
    ]
    base = bass_events(line).codes
    moved = bass_events([replace(n, pitch=n.pitch + 5) for n in line]).codes
    half_time = bass_events(
        [replace(n, start=n.start * 2, duration=n.duration * 2) for n in line]
    ).codes
    double_time = bass_events(
        [replace(n, start=n.start // 2, duration=n.duration // 2) for n in line]
    ).codes
    assert np.array_equal(base, moved)
    assert np.array_equal(base, half_time)
    assert np.array_equal(base, double_time)
    assert len(base) == len(line) - 2


def test_drum_rescale_round_trip() -> None:
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 2**20, size=16)
    assert np.array_equal(rescale(rescale(cells, 2.0, 5), 0.5, 5), cells)


def test_hashes_depend_on_order() -> None:
    a = ngram_hashes(np.array([1, 2, 3, 4], dtype=np.int64), 3)
    b = ngram_hashes(np.array([3, 2, 1, 4], dtype=np.int64), 3)
    assert len(a) == 2
    assert a[0] != b[0]


# --------------------------------------------------------------------- checks


def test_disguised_bass_copy_is_red(corpus_index, tiny_config: BacklineConfig) -> None:  # type: ignore[no-untyped-def]
    index, library, ids = corpus_index
    checker = OriginalityChecker(index, tiny_config)
    source = load_score(library.score_path(ids[3]))
    window = slice_score(source, 4, 8)
    report = checker.check(_disguise(window, semitones=3, stretch=2, edits=(5, 17)))
    bass = next(p for p in report.parts if p.part == "bass")
    assert report.verdict == "red"
    assert bass.verdict == "red"
    top = bass.corpus.sources[0]
    assert top.song_id == ids[3]
    assert top.bars >= tiny_config.originality.red_bars
    assert top.source_bars[0] >= 5
    assert bass.corpus.matched_bars


def test_generic_parts_are_not_plagiarism(corpus_index, tiny_config: BacklineConfig) -> None:  # type: ignore[no-untyped-def]
    index, _, _ = corpus_index
    report = OriginalityChecker(index, tiny_config).check(_generic())
    assert report.verdict != "red"
    assert all(p.corpus.longest_bars < tiny_config.originality.red_bars for p in report.parts)


def test_half_time_drum_copy_is_found(corpus_index, tiny_config: BacklineConfig) -> None:  # type: ignore[no-untyped-def]
    index, library, ids = corpus_index
    source = load_score(library.score_path(ids[7]))
    window = slice_score(source, 0, 4)
    report = OriginalityChecker(index, tiny_config).check(_disguise(window, semitones=0, stretch=2))
    drums = next(p for p in report.parts if p.part == "drums")
    assert drums.corpus.sources
    assert drums.corpus.sources[0].song_id == ids[7]
    assert drums.verdict == "red"


def test_reference_similarity_only_counts_references(
    corpus_index, tiny_config: BacklineConfig
) -> None:  # type: ignore[no-untyped-def]
    index, library, ids = corpus_index
    checker = OriginalityChecker(index, tiny_config)
    copied = slice_score(load_score(library.score_path(ids[3])), 4, 8)
    with_source = checker.check(copied, references=[ids[3]])
    without = checker.check(copied, references=[ids[10]])
    bass_with = next(p for p in with_source.parts if p.part == "bass").references
    bass_without = next(p for p in without.parts if p.part == "bass").references
    assert bass_with is not None
    assert bass_without is not None
    assert bass_with.longest_bars >= 2
    assert all(s.song_id == ids[10] for s in bass_without.sources)


def test_excluding_the_own_group(corpus_index, tiny_config: BacklineConfig) -> None:  # type: ignore[no-untyped-def]
    index, library, ids = corpus_index
    checker = OriginalityChecker(index, tiny_config)
    song = ids[3]
    group = next(s.group for s in index.songs if s.song_id == song)
    window = slice_score(load_score(library.score_path(song)), 4, 8)
    report = checker.check(window, exclude_group=group)
    assert all(s.song_id != song for p in report.parts for s in p.corpus.sources)


def test_calibration_and_storage(corpus_index, tmp_path: Path, tiny_config: BacklineConfig) -> None:  # type: ignore[no-untyped-def]
    index, _, _ = corpus_index
    for part in ("bass", "drums"):
        limits = index.meta["thresholds"][part]
        assert 0 < limits["yellow"] < limits["red"] <= 1.1
        assert index.meta["calibration"][part]["between_songs"]["n"] > 0
    index.save(tmp_path / "copy")
    loaded = OriginalityIndex.load(tmp_path / "copy")
    assert [s.song_id for s in loaded.songs] == [s.song_id for s in index.songs]
    postings = loaded.channels["bass"].postings[tiny_config.originality.seed]
    original = index.channels["bass"].postings[tiny_config.originality.seed]
    assert np.array_equal(postings.hashes, original.hashes)
    assert loaded.meta["thresholds"] == index.meta["thresholds"]


def test_cli_index_and_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: BacklineConfig
) -> None:
    from typer.testing import CliRunner

    from backline.cli import app
    from backline.formats.midi import write_midi

    tiny = str(Path(__file__).resolve().parents[2] / "configs" / "tiny.yaml")
    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BACKLINE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    runner = CliRunner()
    tabs = tmp_path / "tabs"
    tabs.mkdir()
    songs = generate_corpus(tiny_config, n_songs=12, seed=21)
    for index, song in enumerate(songs):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    query = tmp_path / "query.mid"
    write_midi(slice_score(songs[2].score, 0, 4), query)
    assert runner.invoke(app, ["check-originality", str(query), "-c", tiny]).exit_code == 1
    assert runner.invoke(app, ["ingest", str(tabs), "-c", tiny, "--workers", "1"]).exit_code == 0
    built = runner.invoke(app, ["build-originality-index", "-c", tiny])
    assert built.exit_code == 0, built.output
    assert "rare-pattern share" in built.output
    checked = runner.invoke(app, ["check-originality", str(query), "-c", tiny])
    assert checked.exit_code == 0, checked.output
    assert "verdict" in checked.output
    assert "not a legal guarantee" in checked.output


def test_versions_look_like_copies(tmp_path: Path, tiny_config: BacklineConfig) -> None:
    root = tmp_path / "tabs"
    build_library(root, tiny_config)
    library = LibraryPaths(tmp_path / "c.sqlite", tmp_path / "s", tmp_path / "r")
    run_ingest(root, tiny_config, library, workers=1)
    index = build_originality_index(tiny_config, library, tmp_path / "index")
    versions = index.meta["calibration"]["bass"]["versions_of_a_song"]
    between = index.meta["calibration"]["bass"]["between_songs"]
    assert versions["n"] > 0
    assert versions["p50"] >= between["p50"]
