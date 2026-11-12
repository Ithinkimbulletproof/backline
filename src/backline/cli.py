# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Command-line interface of the core package: ``backline --help``."""

from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml

from backline import __version__
from backline.config import BacklineConfig, ConfigError, PathSettings, load_config

app = typer.Typer(
    help="Backline: bass and drums for a guitar riff, built on an explicit harmonic plan.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Profile YAML. Defaults to BACKLINE_CONFIG."),
]


def _load(config: Path | None) -> BacklineConfig:
    try:
        return load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def doctor(
    config: ConfigOption = None,
    require_cuda: Annotated[
        bool, typer.Option(help="Exit with an error when no usable CUDA device is found.")
    ] = False,
) -> None:
    """Check the environment: Python, paths, config and the PyTorch/CUDA build."""
    paths = PathSettings()
    _row("backline", __version__)
    _row("python", f"{platform.python_version()} ({sys.executable})")
    _row("platform", platform.platform())
    cfg = _load(config)
    _row("config", f"{cfg.name} ({config or paths.config})")
    for label, path in (
        ("data", paths.data_dir),
        ("artifacts", paths.artifacts_dir),
        ("checkpoints", paths.checkpoints_dir),
        ("adapters", paths.adapters_dir),
        ("media", paths.media_dir),
    ):
        state = "ok" if path.is_dir() else "missing (created on first use)"
        _row(label, f"{path} [{state}]")

    cuda_ok = _report_torch()
    if require_cuda and not cuda_ok:
        raise typer.Exit(code=1)


def _row(label: str, value: str) -> None:
    typer.echo(f"{label:<12} {value}")


def _report_torch() -> bool:
    try:
        import torch
    except ImportError:
        _row("torch", "not installed: `uv sync --extra cuda` (or `--extra cpu`)")
        return False
    _row("torch", f"{torch.__version__} (CUDA build: {torch.version.cuda or 'none'})")
    if not torch.cuda.is_available():
        _row("cuda", "not available: training will run on the CPU")
        return False
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = arch in torch.cuda.get_arch_list()
    _row("device", f"{torch.cuda.get_device_name(0)} ({arch})")
    _row("kernels", "ok" if supported else f"MISSING for {arch}")
    _row("bf16", "yes" if torch.cuda.is_bf16_supported() else "no")
    return supported


@app.command("config")
def show_config(config: ConfigOption = None) -> None:
    """Print the resolved profile (after ``extends:``) and the data paths."""
    cfg = _load(config)
    data = cfg.model_dump(mode="json")
    typer.echo(yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip())
    typer.echo("# paths (from BACKLINE_* environment variables or .env)")
    paths = PathSettings().model_dump(mode="json")
    typer.echo(yaml.safe_dump(paths, sort_keys=False).rstrip())


_UNSAFE = re.compile(r"[^\w\- ]+")


@app.command()
def synth(
    config: ConfigOption = None,
    out: Annotated[
        Path | None, typer.Option(help="Output folder. Defaults to <data_dir>/synth.")
    ] = None,
    n: Annotated[int | None, typer.Option("--n", min=1, help="Number of songs.")] = None,
    seed: Annotated[int | None, typer.Option(help="Seed. Defaults to the profile seed.")] = None,
    formats: Annotated[str, typer.Option(help="Comma-separated: gp5, mid.")] = "gp5,mid",
) -> None:
    """Generate a synthetic corpus with ground-truth harmony (for tests and demos)."""
    from backline.formats.gp5 import write_gp5
    from backline.formats.midi import write_midi
    from backline.synth import generate_corpus

    wanted = {f.strip().lower() for f in formats.split(",") if f.strip()}
    unknown = wanted - {"gp5", "mid"}
    if unknown:
        raise typer.BadParameter(f"unknown formats: {', '.join(sorted(unknown))}")
    cfg = _load(config)
    target = out or PathSettings().data_dir / "synth"
    target.mkdir(parents=True, exist_ok=True)
    songs = generate_corpus(cfg, n_songs=n, seed=seed)
    for index, song in enumerate(songs):
        stem = _UNSAFE.sub("", f"{index:04d} {song.score.artist} - {song.score.title}")
        if "gp5" in wanted:
            write_gp5(song.score, target / f"{stem}.gp5")
        if "mid" in wanted:
            write_midi(song.score, target / f"{stem}.mid")
        truth = json.dumps(song.truth_dict(), ensure_ascii=False, indent=1)
        (target / f"{stem}.truth.json").write_text(truth, encoding="utf-8")
    typer.echo(f"wrote {len(songs)} songs to {target}")


if __name__ == "__main__":  # pragma: no cover
    app()
