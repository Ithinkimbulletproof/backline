# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from backline.cli import app

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TINY = str(CONFIGS / "tiny.yaml")
runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "config", "synth", "inspect", "ingest", "catalog"):
        assert command in result.output


def test_config_prints_resolved_profile() -> None:
    result = runner.invoke(app, ["config", "-c", TINY])
    assert result.exit_code == 0, result.output
    assert "name: tiny" in result.output
    assert "data_dir:" in result.output


def test_config_reports_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "-c", str(bad)])
    assert result.exit_code == 2


def test_doctor_runs_without_requiring_cuda() -> None:
    result = runner.invoke(app, ["doctor", "-c", TINY])
    assert result.exit_code == 0, result.output
    assert "torch" in result.output


def test_synth_writes_tabs_midi_and_truth(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "-c", TINY, "--n", "2", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("*.gp5"))) == 2
    assert len(list(tmp_path.glob("*.mid"))) == 2
    truths = sorted(tmp_path.glob("*.truth.json"))
    assert len(truths) == 2
    truth = json.loads(truths[0].read_text(encoding="utf-8"))
    assert {"chords", "sections", "bass_notes"} <= truth.keys()


def test_synth_can_write_gp7_containers(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["synth", "-c", TINY, "--n", "1", "--out", str(tmp_path), "--formats", "gp"]
    )
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("*.gp"))) == 1
    assert not list(tmp_path.glob("*.gp5"))


def test_inspect_shows_tracks_and_notes(tmp_path: Path) -> None:
    runner.invoke(app, ["synth", "-c", TINY, "--n", "1", "--out", str(tmp_path), "--formats", "gp"])
    (path,) = tmp_path.glob("*.gp")
    result = runner.invoke(app, ["inspect", str(path), "--track", "2", "--bars", "1"])
    assert result.exit_code == 0, result.output
    assert "Guitar Pro 7+" in result.output
    assert "drumKit" in result.output
    assert "notes of track 2" in result.output
    assert "Kick" in result.output or "Crash" in result.output


def test_inspect_reports_unreadable_files(tmp_path: Path) -> None:
    path = tmp_path / "broken.gp"
    path.write_bytes(b"nope")
    result = runner.invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1


def test_inspect_rejects_bad_track(tmp_path: Path) -> None:
    runner.invoke(
        app, ["synth", "-c", TINY, "--n", "1", "--out", str(tmp_path), "--formats", "mid"]
    )
    (path,) = tmp_path.glob("*.mid")
    result = runner.invoke(app, ["inspect", str(path), "--track", "9"])
    assert result.exit_code != 0


def test_ingest_and_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backline.config import load_config
    from tests.library import build_library

    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BACKLINE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    root = tmp_path / "tabs"
    build_library(root, load_config(TINY))

    result = runner.invoke(app, ["ingest", str(root), "-c", TINY, "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert "status       duplicate=4" in result.output
    listing = runner.invoke(app, ["catalog", "--status", "filtered"])
    assert listing.exit_code == 0, listing.output
    assert "no bass track" in listing.output


def test_catalog_without_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "data"))
    assert runner.invoke(app, ["catalog"]).exit_code == 1


def test_ingest_missing_folder(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope"), "-c", TINY])
    assert result.exit_code == 1


def test_synth_rejects_unknown_formats(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "-c", TINY, "--out", str(tmp_path), "--formats", "pdf"])
    assert result.exit_code != 0
