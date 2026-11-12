# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

import pytest
from typer.testing import CliRunner

from backline.cli import app
from backline.config import BacklineConfig
from backline.formats.gpif_writer import write_gp
from backline.ingest.pipeline import LibraryPaths, run_ingest
from backline.synth import generate_corpus
from backline.theory.annotate import load_annotation
from backline.theory.pipeline import annotate_library
from tests.library import build_library

TINY = str(Path(__file__).resolve().parents[2] / "configs" / "tiny.yaml")
runner = CliRunner()


def test_library_annotation_and_corpus_stats(tmp_path: Path, tiny_config: BacklineConfig) -> None:
    root = tmp_path / "tabs"
    build_library(root, tiny_config)
    library = LibraryPaths(tmp_path / "catalog.sqlite", tmp_path / "scores", tmp_path / "reports")
    run_ingest(root, tiny_config, library, workers=1)

    first = annotate_library(library)
    assert first.annotated == 14
    assert first.harmony.songs == 10
    assert 0 < first.harmony.power_chord_share < 1
    assert abs(sum(first.harmony.functions.values()) - 1) < 1e-3
    assert first.harmony.bass_roles
    assert Path(first.report_path).is_file()

    again = annotate_library(library)
    assert (again.annotated, again.reused) == (0, 14)
    assert again.harmony == first.harmony

    song_id = next(p.stem for p in (tmp_path / "scores").glob("*.json.gz")).split(".")[0]
    assert load_annotation(library.annotation_path(song_id)).keys


def test_cli_annotates_a_single_file(tmp_path: Path, tiny_config: BacklineConfig) -> None:
    path = tmp_path / "song.gp"
    write_gp(generate_corpus(tiny_config, n_songs=1, seed=5)[0].score, path)
    result = runner.invoke(app, ["annotate", "--file", str(path), "--bars", "1:4", "-c", TINY])
    assert result.exit_code == 0, result.output
    assert "functions" in result.output
    assert result.output.count("\n") >= 6


def test_cli_eval_theory() -> None:
    result = runner.invoke(app, ["eval-theory", "--n", "5", "-c", TINY])
    assert result.exit_code == 0, result.output
    assert "chord root" in result.output


def test_cli_annotate_needs_a_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "data"))
    assert runner.invoke(app, ["annotate", "-c", TINY]).exit_code == 1
