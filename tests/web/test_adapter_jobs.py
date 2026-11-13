# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Adapter jobs: a child process trains the adapter while the job row tracks progress."""

from __future__ import annotations

from pathlib import Path

import pytest
from django.test import Client
from typer.testing import CliRunner

from backline.cli import app
from backline.config import BacklineConfig
from backline.formats.guitarpro import write_gp5
from backline.ingest.catalog import Catalog
from backline.synth import generate_corpus
from web.studio.jobs import adapter_command, run_adapter_job
from web.studio.models import AdapterJob

TINY = str(Path(__file__).resolve().parents[2] / "configs" / "tiny.yaml")


@pytest.fixture
def trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: BacklineConfig
) -> list[str]:
    for name, folder in (
        ("DATA", "data"),
        ("ARTIFACTS", "artifacts"),
        ("CHECKPOINTS", "ckpt"),
        ("ADAPTERS", "adapters"),
    ):
        monkeypatch.setenv(f"BACKLINE_{name}_DIR", str(tmp_path / folder))
    monkeypatch.setenv("BACKLINE_CONFIG", TINY)
    tabs = tmp_path / "tabs"
    tabs.mkdir()
    for index, song in enumerate(generate_corpus(tiny_config, n_songs=10, seed=13)):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    runner = CliRunner()
    for command in (
        ["ingest", str(tabs), "--workers", "1"],
        ["build-dataset"],
        ["train", "--device", "cpu"],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    with Catalog(tmp_path / "data" / "catalog.sqlite") as catalog:
        return [s.id for s in catalog.songs("ok")][:2]


def test_command_line(tmp_path: Path) -> None:
    command = adapter_command(["a", "b"], tmp_path / "p.jsonl")
    assert command[1:5] == ["-m", "backline.cli", "adapter", "train"]
    assert command.count("--ref") == 2


@pytest.mark.django_db
def test_job_trains_then_hits_the_cache(trained: list[str]) -> None:
    job = AdapterJob.objects.create(references=trained)
    finished = run_adapter_job(job.pk, poll=0.2)
    assert finished.status == AdapterJob.Status.DONE, finished.message
    assert finished.progress == 1.0
    assert len(finished.key) == 20

    again = run_adapter_job(
        AdapterJob.objects.create(references=list(reversed(trained))).pk, poll=0.2
    )
    assert again.status == AdapterJob.Status.CACHED
    assert again.key == finished.key


@pytest.mark.django_db
def test_failed_job_reports_the_error(trained: list[str]) -> None:
    job = run_adapter_job(AdapterJob.objects.create(references=["missing"]).pk, poll=0.2)
    assert job.status == AdapterJob.Status.FAILED
    assert "segments" in job.message


@pytest.mark.django_db
def test_views_queue_and_report(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[int] = []
    monkeypatch.setattr("web.studio.views.train_adapter_task", queued.append)
    response = client.post("/adapters/", {"ref": ["b", "a"]})
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert queued == [job_id]
    assert AdapterJob.objects.get(pk=job_id).references == ["a", "b"]
    status = client.get(f"/jobs/{job_id}/").json()
    assert status["status"] == "queued"
    assert status["label"] == "в очереди"
    assert client.post("/adapters/", {}).status_code == 400
