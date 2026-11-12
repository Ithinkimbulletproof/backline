# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from backline.cli import app
from backline.config import BacklineConfig
from backline.dataset.build import build_dataset
from backline.dataset.reader import SegmentDataset
from backline.formats.guitarpro import write_gp5
from backline.ingest.catalog import Catalog
from backline.ingest.pipeline import LibraryPaths, run_ingest
from backline.synth import generate_corpus
from backline.tokenizer.encode import decode

TINY = str(Path(__file__).resolve().parents[2] / "configs" / "tiny.yaml")


@pytest.fixture(scope="module")
def built(
    tmp_path_factory: pytest.TempPathFactory, tiny_config: BacklineConfig
) -> tuple[Path, LibraryPaths]:
    root = tmp_path_factory.mktemp("lib")
    tabs = root / "tabs"
    tabs.mkdir()
    for index, song in enumerate(generate_corpus(tiny_config, n_songs=30, seed=77)):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    library = LibraryPaths(root / "catalog.sqlite", root / "scores", root / "reports")
    config = tiny_config.model_copy(
        update={"ingest": tiny_config.ingest.model_copy(update={"val_fraction": 0.3})}
    )
    run_ingest(tabs, config, library, workers=1)
    out = root / "dataset"
    build_dataset(tiny_config, library, out)
    return out, library


def test_files_and_report(built: tuple[Path, LibraryPaths]) -> None:
    out, _ = built
    for name in ("vocab.json", "meta.json", "songs.json", "train.tokens", "train.index.npy"):
        assert (out / name).is_file()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    report = meta["report"]
    assert report["segments"]["train"] > 50
    assert report["segments"]["val"] > 0
    assert report["length_percentiles"]["max"] <= meta["context_length"]


def test_splits_never_share_songs(
    built: tuple[Path, LibraryPaths], tiny_config: BacklineConfig
) -> None:
    out, library = built
    train = SegmentDataset(out, "train", tiny_config)
    val = SegmentDataset(out, "val", tiny_config)
    train_songs = {train.song_ids[int(s)] for s in train.index["song"]}
    val_songs = {val.song_ids[int(s)] for s in val.index["song"]}
    assert train_songs
    assert val_songs
    assert not train_songs & val_songs
    with Catalog(library.catalog) as catalog:
        splits = {s.id: s.split for s in catalog.songs()}
    assert {splits[s] for s in val_songs} == {"val"}


def test_samples_mark_the_target(
    built: tuple[Path, LibraryPaths], tiny_config: BacklineConfig
) -> None:
    out, _ = built
    dataset = SegmentDataset(out, "train", tiny_config)
    for i in range(0, len(dataset), max(len(dataset) // 10, 1)):
        sample = dataset[i]
        assert dataset.vocab.token(int(sample.tokens[0])) == "BOS"
        assert dataset.vocab.token(int(sample.tokens[sample.target_start])) == "PLAN"
        assert dataset.vocab.token(int(sample.tokens[sample.target_start - 1])) == "SEP"
        assert dataset.vocab.token(int(sample.tokens[-1])) == "EOS"


def test_augmentation_transposes_pitches_only(
    built: tuple[Path, LibraryPaths], tiny_config: BacklineConfig
) -> None:
    out, _ = built
    plain = SegmentDataset(out, "train", tiny_config)
    augmented = SegmentDataset(out, "train", tiny_config, augment=True, seed=3)
    shifts = set()
    for i in range(min(len(plain), 40)):
        a, b = plain[i], augmented[i]
        shifts.add(b.shift)
        before, after = (
            decode(a.tokens.tolist(), plain.vocab),
            decode(b.tokens.tolist(), plain.vocab),
        )
        assert [n.pitch + b.shift for n in before.bass] == [n.pitch for n in after.bass]
        assert after.drums == before.drums
        assert len(a.tokens) == len(b.tokens)
    assert len(shifts) > 1


def test_vocabulary_mismatch_is_detected(
    built: tuple[Path, LibraryPaths], tiny_config: BacklineConfig
) -> None:
    out, _ = built
    other = tiny_config.model_copy(
        update={"tokenizer": tiny_config.tokenizer.model_copy(update={"max_positions": 96})}
    )
    with pytest.raises(ValueError, match="different vocabulary"):
        SegmentDataset(out, "train", other)


def test_long_segments_are_dropped(
    built: tuple[Path, LibraryPaths], tiny_config: BacklineConfig, tmp_path: Path
) -> None:
    _, library = built
    short = tiny_config.model_copy(
        update={"model": tiny_config.model.model_copy(update={"context_length": 200})}
    )
    report = build_dataset(short, library, tmp_path / "short")
    assert report.dropped.get("longer_than_context", 0) > 0
    index = np.load(tmp_path / "short" / "train.index.npy")
    assert (index["length"] <= 200).all()


def test_cli_build_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: BacklineConfig
) -> None:
    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BACKLINE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    runner = CliRunner()
    assert runner.invoke(app, ["build-dataset", "-c", TINY]).exit_code == 1
    tabs = tmp_path / "tabs"
    tabs.mkdir()
    for index, song in enumerate(generate_corpus(tiny_config, n_songs=4, seed=5)):
        write_gp5(song.score, tabs / f"{index} {song.score.artist} - {song.score.title}.gp5")
    assert runner.invoke(app, ["ingest", str(tabs), "-c", TINY, "--workers", "1"]).exit_code == 0
    result = runner.invoke(app, ["build-dataset", "-c", TINY, "--name", "demo"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "artifacts" / "datasets" / "demo" / "train.tokens").is_file()
