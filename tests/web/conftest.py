# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""A trained workspace shared by the web tests (synthetic library, tiny model)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from backline.cli import app
from backline.config import BacklineConfig
from backline.formats.guitarpro import write_gp5
from backline.synth import generate_corpus
from web.studio import services

TINY = str(Path(__file__).resolve().parents[2] / "configs" / "tiny.yaml")
FOLDERS = (
    ("DATA", "data"),
    ("ARTIFACTS", "artifacts"),
    ("CHECKPOINTS", "ckpt"),
    ("ADAPTERS", "adapters"),
)


def _point(root: Path, mp: pytest.MonkeyPatch) -> None:
    """Point the core and the studio at the folders of ``root``."""
    for name, folder in FOLDERS:
        mp.setenv(f"BACKLINE_{name}_DIR", str(root / folder))
    mp.setenv("BACKLINE_CONFIG", TINY)
    services.config.cache_clear()
    services._loaded.clear()


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory, tiny_config: BacklineConfig) -> Path:
    """A synthetic library with a dataset, an originality index and a trained tiny model."""
    root = tmp_path_factory.mktemp("workspace")
    tabs = root / "tabs"
    tabs.mkdir()
    for index, song in enumerate(generate_corpus(tiny_config, n_songs=10, seed=29)):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    with pytest.MonkeyPatch.context() as mp:
        _point(root, mp)
        runner = CliRunner()
        for command in (
            ["ingest", str(tabs), "--workers", "1"],
            ["build-dataset"],
            ["build-originality-index"],
            ["train", "--device", "cpu"],
        ):
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def use_workspace(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Path], None]]:
    """Point the app at a workspace for one test (the model cache is dropped around it)."""
    yield lambda root: _point(root, monkeypatch)
    services.config.cache_clear()
    services._loaded.clear()
