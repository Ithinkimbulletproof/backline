# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

from typer.testing import CliRunner

from backline.cli import app

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TINY = str(CONFIGS / "tiny.yaml")
runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "config", "synth"):
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


def test_synth_rejects_unknown_formats(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "-c", TINY, "--out", str(tmp_path), "--formats", "pdf"])
    assert result.exit_code != 0
