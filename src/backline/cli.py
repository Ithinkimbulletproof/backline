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
    formats: Annotated[
        str,
        typer.Option(help="Comma-separated: gp5, gp (minimal GP7 container), mid."),
    ] = "gp5,mid",
) -> None:
    """Generate a synthetic corpus with ground-truth harmony (for tests and demos)."""
    from backline.formats.gpif_writer import write_gp
    from backline.formats.guitarpro import write_gp5
    from backline.formats.midi import write_midi
    from backline.synth import generate_corpus

    writers = {"gp5": write_gp5, "gp": write_gp, "mid": write_midi}
    wanted = [f.strip().lower() for f in formats.split(",") if f.strip()]
    unknown = sorted(set(wanted) - set(writers))
    if unknown:
        raise typer.BadParameter(f"unknown formats: {', '.join(unknown)}")
    cfg = _load(config)
    target = out or PathSettings().data_dir / "synth"
    target.mkdir(parents=True, exist_ok=True)
    songs = generate_corpus(cfg, n_songs=n, seed=seed)
    for index, song in enumerate(songs):
        stem = _UNSAFE.sub("", f"{index:04d} {song.score.artist} - {song.score.title}")
        for suffix in wanted:
            writers[suffix](song.score, target / f"{stem}.{suffix}")
        truth = json.dumps(song.truth_dict(), ensure_ascii=False, indent=1)
        (target / f"{stem}.truth.json").write_text(truth, encoding="utf-8")
    typer.echo(f"wrote {len(songs)} songs to {target}")


def _track_row(
    index: str,
    name: str,
    role: str,
    instrument: str,
    program: str,
    tuning: str,
    notes: str,
    span: str,
) -> str:
    return (
        f"  {index:>2}  {name[:22]:<22} {role:<7} {instrument[:15]:<15} {program:>4}  "
        f"{tuning[:20]:<20} {notes:>6}  {span}"
    )


def _parse_bars(spec: str | None, n_bars: int) -> range:
    """Parse a 1-based inclusive bar range such as ``3``, ``3:6`` or ``3-6``."""
    if not spec:
        return range(0, min(n_bars, 2))
    first, _, last = spec.replace("-", ":").partition(":")
    try:
        lo = int(first)
        hi = int(last) if last else lo
    except ValueError as exc:
        raise typer.BadParameter(f"invalid bar range: {spec!r}") from exc
    if not 1 <= lo <= hi:
        raise typer.BadParameter(f"invalid bar range: {spec!r}")
    return range(lo - 1, min(hi, n_bars))


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Tab or MIDI file.")],
    track: Annotated[
        int | None, typer.Option("--track", "-t", help="Also list the notes of this track.")
    ] = None,
    bars: Annotated[
        str | None, typer.Option(help="Bars to list with --track, 1-based: 5 or 5:8.")
    ] = None,
    no_feel: Annotated[
        bool, typer.Option("--no-feel", help="Keep notation timing; do not apply triplet feel.")
    ] = False,
) -> None:
    """Show what Backline reads from a file: meters, tempos, tracks and, optionally, notes."""
    from backline.drums import GM_PERCUSSION_NAMES
    from backline.formats.common import ReadStats, ScoreReadError
    from backline.formats.reader import FORMAT_NAMES, read_score
    from backline.formats.tablature import duration_label
    from backline.music import pitch_name

    stats = ReadStats()
    try:
        score = read_score(path, stats=stats, apply_triplet_feel=not no_feel)
    except ScoreReadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    starts = score.bar_starts()
    _row("file", str(path))
    _row("format", FORMAT_NAMES.get(path.suffix.lower(), path.suffix))
    _row("title", score.title or "-")
    _row("artist", score.artist or "-")
    _row("bars", str(score.n_bars))
    _row("meter", ", ".join(f"{ts} @bar {ts.bar + 1}" for ts in score.time_signatures))
    _row(
        "tempo",
        ", ".join(
            f"{t.bpm:g} @bar {score.bar_of(min(t.tick, starts[-1] - 1)) + 1}" for t in score.tempos
        ),
    )
    typer.echo("tracks")
    columns = ("#", "name", "role", "instrument", "prog", "tuning", "notes", "range")
    typer.echo(_track_row(*columns))
    for index, item in enumerate(score.tracks):
        tuning = " ".join(pitch_name(p) for p in item.tuning) if item.tuning else "-"
        pitches = [n.pitch for n in item.notes]
        if not pitches:
            span = "-"
        elif item.is_percussion:
            span = f"{len(set(pitches))} drum notes"
        else:
            span = f"{pitch_name(min(pitches))}-{pitch_name(max(pitches))}"
        typer.echo(
            _track_row(
                str(index),
                item.name,
                item.role.value,
                item.instrument or "-",
                str(item.program),
                tuning,
                str(len(item.notes)),
                span,
            )
        )
    if stats:
        _row("skipped", ", ".join(f"{key}={value}" for key, value in sorted(stats.items())))

    if track is None:
        return
    if not 0 <= track < len(score.tracks):
        raise typer.BadParameter(f"track must be within 0..{len(score.tracks) - 1}")
    chosen = score.tracks[track]
    selected = _parse_bars(bars, score.n_bars)
    typer.echo(f"notes of track {track} ({chosen.name}), bars {selected.start + 1}-{selected.stop}")
    for note in chosen.sorted_notes():
        bar = score.bar_of(note.start)
        if bar not in selected:
            continue
        beat_ticks = score.time_signature_at(bar).beat_ticks(score.tpq)
        beat, tick = divmod(note.start - starts[bar], beat_ticks)
        if chosen.is_percussion:
            name = chosen.percussion_names.get(note.pitch) or GM_PERCUSSION_NAMES.get(
                note.pitch, f"drum {note.pitch}"
            )
        else:
            name = pitch_name(note.pitch)
        flags = " ".join((f.name or "").lower() for f in type(note.flags) if f and f in note.flags)
        typer.echo(
            f"  {bar + 1:>4}.{beat + 1}.{tick:03d}  {name:<18} {duration_label(note.duration):<10}"
            f" vel {note.velocity:<3} {flags}".rstrip()
        )


@app.command()
def ingest(
    source: Annotated[
        Path | None,
        typer.Argument(help="Tab library folder. Defaults to BACKLINE_TABS_DIR."),
    ] = None,
    config: ConfigOption = None,
    workers: Annotated[
        int | None, typer.Option(min=0, help="Reader processes (0 = one per core).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-read files even if they did not change.")] = False,
) -> None:
    """Import a tab library: roles, filtering, duplicates, split, catalog and report."""
    from backline.ingest.pipeline import LibraryPaths, run_ingest

    cfg = _load(config)
    paths = PathSettings()
    folder = source or paths.tabs_dir
    step = 0

    def progress(done: int, total: int) -> None:
        nonlocal step
        if done == total or done // 50 > step:
            step = done // 50
            typer.echo(f"read {done}/{total}", err=True)

    try:
        report = run_ingest(
            folder,
            cfg,
            LibraryPaths.from_settings(paths),
            workers=workers,
            force=force,
            progress=progress,
        )
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in report.lines():
        typer.echo(line)


@app.command()
def catalog(
    status: Annotated[
        str | None, typer.Option(help="Only songs with this status (ok, duplicate, filtered...).")
    ] = None,
    limit: Annotated[int, typer.Option(min=0, help="How many songs to list.")] = 30,
) -> None:
    """Summarise the imported library."""
    from collections import Counter

    from backline.ingest.catalog import Catalog
    from backline.ingest.pipeline import LibraryPaths

    library = LibraryPaths.from_settings(PathSettings())
    if not library.catalog.is_file():
        typer.echo("no catalog yet: run `backline ingest` first", err=True)
        raise typer.Exit(code=1)
    with Catalog(library.catalog) as db:
        songs = db.songs()
        source = db.get_meta("source") or "-"
    _row("library", source)
    _row("songs", str(len(songs)))
    _row(
        "status", ", ".join(f"{k}={v}" for k, v in sorted(Counter(s.status for s in songs).items()))
    )
    splits = Counter(s.split for s in songs if s.status == "ok")
    _row("splits", ", ".join(f"{k}={v}" for k, v in sorted(splits.items())) or "-")
    shown = [s for s in songs if status is None or s.status == status][:limit]
    for song in shown:
        name = f"{song.artist or '?'} - {song.title or '?'}"
        detail = song.split or song.reason
        typer.echo(f"  {song.status:<11} {name[:50]:<50} {detail}")


@app.command()
def annotate(
    file: Annotated[
        Path | None,
        typer.Option(
            "--file", "-f", exists=True, dir_okay=False, help="Analyse one tab or MIDI file."
        ),
    ] = None,
    bars: Annotated[str | None, typer.Option(help="Bars to print with --file, e.g. 1:16.")] = None,
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option(help="Re-annotate songs that did not change.")] = False,
) -> None:
    """Harmonic analysis: keys, chords per beat, functions and bass note roles.

    Without --file, annotates the whole imported library and prints corpus statistics.
    """
    from backline.ingest.pipeline import LibraryPaths

    cfg = _load(config)
    if file is None:
        from backline.theory.pipeline import annotate_library

        library = LibraryPaths.from_settings(PathSettings())
        if not library.catalog.is_file():
            typer.echo("no catalog yet: run `backline ingest` first", err=True)
            raise typer.Exit(code=1)
        for line in annotate_library(library, force=force).lines():
            typer.echo(line)
        return

    from backline.formats.common import ScoreReadError
    from backline.formats.reader import read_score
    from backline.ingest.roles import assign_roles
    from backline.theory.annotate import annotate_score
    from backline.theory.table import annotation_table

    try:
        score = assign_roles(read_score(file), cfg.ingest.name_hints)
    except ScoreReadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    selected = _parse_bars(bars, score.n_bars) if bars else None
    for line in annotation_table(score, annotate_score(score), selected):
        typer.echo(line)


@app.command("build-dataset")
def build_dataset_command(
    config: ConfigOption = None,
    name: Annotated[
        str | None, typer.Option(help="Dataset folder name. Defaults to the profile name.")
    ] = None,
) -> None:
    """Tokenise the imported, annotated library into train/val/test shards."""
    from backline.dataset.build import build_dataset, dataset_dir
    from backline.ingest.pipeline import LibraryPaths

    cfg = _load(config)
    paths = PathSettings()
    library = LibraryPaths.from_settings(paths)
    if not library.catalog.is_file():
        typer.echo("no catalog yet: run `backline ingest` first", err=True)
        raise typer.Exit(code=1)
    report = build_dataset(cfg, library, dataset_dir(paths.artifacts_dir, name or cfg.name))
    for line in report.lines():
        typer.echo(line)


@app.command("build-originality-index")
def build_originality_index_command(config: ConfigOption = None) -> None:
    """Index the imported library for copy detection and calibrate the thresholds."""
    from backline.ingest.pipeline import LibraryPaths
    from backline.originality.build import build_originality_index

    cfg = _load(config)
    paths = PathSettings()
    library = LibraryPaths.from_settings(paths)
    if not library.catalog.is_file():
        typer.echo("no catalog yet: run `backline ingest` first", err=True)
        raise typer.Exit(code=1)
    target = paths.artifacts_dir / "originality" / cfg.name
    index = build_originality_index(cfg, library, target)
    _row("index", f"{target} ({len(index.songs)} songs, {index.n_groups} groups)")
    for part, limits in index.meta["thresholds"].items():
        _row(
            part, f"yellow >= {limits['yellow']:.0%}, red >= {limits['red']:.0%} rare-pattern share"
        )


@app.command("check-originality")
def check_originality(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Tab or MIDI file.")],
    config: ConfigOption = None,
    reference: Annotated[
        list[str] | None, typer.Option("--ref", help="Song id of a reference (repeatable).")
    ] = None,
) -> None:
    """Check the bass and drums of a file against the corpus (a heuristic, not a legal verdict)."""
    from backline.formats.common import ScoreReadError
    from backline.formats.reader import read_score
    from backline.ingest.roles import assign_roles
    from backline.originality.check import OriginalityChecker
    from backline.originality.index import OriginalityIndex

    cfg = _load(config)
    folder = PathSettings().artifacts_dir / "originality" / cfg.name
    if not (folder / "index.npz").is_file():
        typer.echo("no originality index yet: run `backline build-originality-index`", err=True)
        raise typer.Exit(code=1)
    try:
        score = assign_roles(read_score(file), cfg.ingest.name_hints)
    except ScoreReadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    report = OriginalityChecker(OriginalityIndex.load(folder), cfg).check(
        score, references=reference or []
    )
    _row("verdict", f"{report.verdict} (score {report.score}/100)")
    for part in report.parts:
        sim = part.corpus
        _row(
            part.part,
            f"{part.verdict}: longest {sim.longest_bars:.1f} bars, rare share {sim.rare_share:.0%}",
        )
        for source in sim.sources[:3]:
            typer.echo(
                f"{'':13}like {source.artist} - {source.title}, bars {source.source_bars[0]}-"
                f"{source.source_bars[1]} ({source.bars:.1f} bars)"
            )
        for reason in part.reasons:
            typer.echo(f"{'':13}{reason}")
    typer.echo("note: a heuristic estimate of copying, not a legal guarantee")


@app.command()
def train(
    config: ConfigOption = None,
    dataset: Annotated[
        str | None, typer.Option(help="Dataset folder name. Defaults to the profile name.")
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="cuda or cpu (default: cuda if available).")
    ] = None,
    fresh: Annotated[bool, typer.Option(help="Start over instead of resuming last.pt.")] = False,
) -> None:
    """Train the base model on a built dataset (resumes automatically)."""
    from backline.dataset.build import dataset_dir
    from backline.model.train import Trainer
    from backline.originality.index import OriginalityIndex

    cfg = _load(config)
    paths = PathSettings()
    data = dataset_dir(paths.artifacts_dir, dataset or cfg.name)
    if not (data / "train.tokens").is_file():
        typer.echo(f"no dataset at {data}: run `backline build-dataset` first", err=True)
        raise typer.Exit(code=1)
    index_dir = paths.artifacts_dir / "originality" / cfg.name
    index = OriginalityIndex.load(index_dir) if (index_dir / "index.npz").is_file() else None
    if index is None:
        typer.echo("note: no originality index, memorization will not be measured", err=True)
    trainer = Trainer(cfg, data, paths.checkpoints_dir / cfg.name, device=device, index=index)
    _row("device", str(trainer.device))
    _row("parameters", f"{trainer.model.parameter_count() / 1e6:.1f}M")

    def progress(step: int, total: int, record: dict[str, float]) -> None:
        if "val_loss" in record or step % 50 == 0:
            shown = ", ".join(f"{k}={v:.4g}" for k, v in record.items() if k != "step")
            typer.echo(f"step {step}/{total}: {shown}")

    summary = trainer.train(resume=not fresh, progress=progress)
    _row("best", f"step {summary.best_step} (metric {summary.best_metric:.4f})")
    _row("checkpoints", str(paths.checkpoints_dir / cfg.name))


def _plan_line(decoded_plan: list[object], bar: int) -> str:
    from backline.music import chord_symbol, prefers_flats

    names = []
    for chord in decoded_plan:
        if getattr(chord, "bar", -1) != bar:
            continue
        root, quality = chord.root, chord.quality  # type: ignore[attr-defined]
        if root is None or quality is None:
            names.append("N.C.")
        else:
            names.append(chord_symbol(root, quality, flats=prefers_flats(chord.tonic, chord.mode)))  # type: ignore[attr-defined]
    return " ".join(names)


@app.command()
def generate(
    riff: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Riff: MIDI or Guitar Pro.")
    ],
    config: ConfigOption = None,
    checkpoint: Annotated[
        Path | None, typer.Option(help="Model checkpoint (default: best.pt).")
    ] = None,
    bars: Annotated[str | None, typer.Option(help="Bars of the riff, e.g. 1:4.")] = None,
    track: Annotated[int | None, typer.Option(help="Track index of the riff guitar.")] = None,
    n: Annotated[int | None, typer.Option("--n", min=1, help="Number of variants.")] = None,
    out: Annotated[Path | None, typer.Option(help="Folder for the MIDI files.")] = None,
    until_green: Annotated[
        bool, typer.Option(help="Regenerate variants until the check is green.")
    ] = False,
    seed: Annotated[int | None, typer.Option(help="Sampling seed.")] = None,
    device: Annotated[str | None, typer.Option(help="cuda or cpu.")] = None,
    reference: ReferenceOption = None,
    strength: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help="Adapter strength (default: the recommended one)."),
    ] = None,
) -> None:
    """Write bass and drums for a riff: a harmonic plan first, then the parts.

    With --ref, the cached adapter for those references leans the parts towards them
    (train it first with `backline adapter train`).
    """
    from backline.adapters.cache import AdapterStore
    from backline.adapters.train import load_for_generation
    from backline.formats.common import ScoreReadError
    from backline.formats.reader import read_score
    from backline.model.service import export_variant, generate_variants, prepare_riff
    from backline.model.train import load_model
    from backline.originality.check import OriginalityChecker
    from backline.originality.index import OriginalityIndex
    from backline.theory.table import key_name
    from backline.tokenizer.vocab import Vocab

    cfg = _load(config)
    paths = PathSettings()
    path = checkpoint or paths.checkpoints_dir / cfg.name / "best.pt"
    if not path.is_file():
        typer.echo(f"no model at {path}: run `backline train` first", err=True)
        raise typer.Exit(code=1)
    _, state = load_model(path, device="cpu")
    vocab = Vocab.from_config(cfg)
    if state["vocab"] != vocab.fingerprint():
        typer.echo("the checkpoint was trained with a different vocabulary", err=True)
        raise typer.Exit(code=1)
    refs = reference or []
    model, adapter = load_for_generation(
        cfg, path, refs, AdapterStore(paths.adapters_dir), device=device
    )
    if refs and adapter is None:
        typer.echo(
            "no adapter for these references yet: run `backline adapter train` first", err=True
        )
        raise typer.Exit(code=1)
    chosen_strength = (
        strength if strength is not None else (adapter.recommended_strength if adapter else 1.0)
    )
    try:
        score = read_score(riff)
    except ScoreReadError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    prepared = prepare_riff(
        score, cfg, bars=_parse_bars(bars, score.n_bars) if bars else None, track=track
    )
    index_dir = paths.artifacts_dir / "originality" / cfg.name
    checker = (
        OriginalityChecker(OriginalityIndex.load(index_dir), cfg)
        if (index_dir / "index.npz").is_file()
        else None
    )
    result = generate_variants(
        model,
        vocab,
        cfg,
        prepared,
        variants=n,
        checker=checker,
        until_green=until_green,
        seed=seed,
        drum_map=cfg.drums.load_output_map(),
        references=refs,
        strength=chosen_strength,
    )
    if adapter is not None:
        _row("adapter", f"{adapter.key}, strength {chosen_strength:g}")
    _row(
        "riff",
        f"{riff.name}, bars {prepared.bars.start + 1}-{prepared.bars.stop}, "
        f"{key_name(prepared.key)}",
    )
    target = out or paths.media_dir / "generated" / riff.stem
    for number, variant in enumerate(result.variants, start=1):
        typer.echo(f"variant {number}")
        for bar in range(prepared.n_bars):
            typer.echo(f"  bar {bar + 1:>2}: {_plan_line(list(variant.decoded.plan), bar)}")
        m = variant.metrics
        _row(
            "  metrics",
            f"plan adherence {m.plan_adherence:.0%}, bass-kick {m.bass_kick_coupling:.0%}",
        )
        if variant.originality is not None:
            report = variant.originality
            _row(
                "  originality",
                f"{report.verdict} ({report.score}/100, attempts {variant.attempts})",
            )
        files = export_variant(variant, target, f"variant_{number}")
        _row("  files", ", ".join(p.name for p in files))
    _row("folder", str(target))


adapter_app = typer.Typer(help="LoRA adapters for sets of reference songs.", no_args_is_help=True)
app.add_typer(adapter_app, name="adapter")

ReferenceOption = Annotated[
    list[str] | None, typer.Option("--ref", help="Song id of a reference (repeatable).")
]


@adapter_app.command("train")
def adapter_train(
    reference: ReferenceOption = None,
    config: ConfigOption = None,
    checkpoint: Annotated[
        Path | None, typer.Option(help="Base checkpoint (default: best.pt).")
    ] = None,
    dataset: Annotated[
        str | None, typer.Option(help="Dataset name (default: profile name).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="cuda or cpu.")] = None,
    force: Annotated[bool, typer.Option(help="Retrain even if the adapter is cached.")] = False,
    progress_file: Annotated[
        Path | None, typer.Option(help="Append progress as JSON lines (used by the web UI).")
    ] = None,
) -> None:
    """Train (or fetch from the cache) the adapter for a set of reference songs."""
    from backline.adapters.cache import AdapterStore
    from backline.adapters.train import AdapterError, train_adapter
    from backline.dataset.build import dataset_dir
    from backline.originality.index import OriginalityIndex

    cfg = _load(config)
    paths = PathSettings()
    base = checkpoint or paths.checkpoints_dir / cfg.name / "best.pt"
    if not base.is_file():
        typer.echo(f"no model at {base}: run `backline train` first", err=True)
        raise typer.Exit(code=1)
    index_dir = paths.artifacts_dir / "originality" / cfg.name
    index = OriginalityIndex.load(index_dir) if (index_dir / "index.npz").is_file() else None

    def report(step: int, total: int, record: dict[str, float]) -> None:
        if progress_file is not None:
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "total": total, **record}) + "\n")

    try:
        result = train_adapter(
            cfg,
            base,
            dataset_dir(paths.artifacts_dir, dataset or cfg.name),
            reference or [],
            AdapterStore(paths.adapters_dir),
            index=index,
            device=device,
            force=force,
            progress=report,
        )
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        if progress_file is not None:
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"error": str(exc)}) + "\n")
        raise typer.Exit(code=1) from exc
    info = result.info
    if progress_file is not None:
        with progress_file.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"done": True, "key": info.key, "cached": result.cached}) + "\n"
            )
    _row("adapter", f"{info.key} ({'from cache' if result.cached else 'trained'})")
    _row("references", ", ".join(info.references))
    _row("share", f"{info.reference_share:.0%} of each batch, {info.steps} steps")
    _row("strength", f"recommended {info.recommended_strength:g}")


@adapter_app.command("list")
def adapter_list() -> None:
    """Cached adapters, newest first."""
    from backline.adapters.cache import AdapterStore

    adapters = AdapterStore(PathSettings().adapters_dir).list()
    if not adapters:
        typer.echo("no adapters yet")
        return
    for info in adapters:
        typer.echo(
            f"{info.key}  {len(info.references)} refs  "
            f"strength {info.recommended_strength:g}  {info.created}"
        )


@app.command("eval-theory")
def eval_theory(
    config: ConfigOption = None,
    n: Annotated[int, typer.Option("--n", min=1, help="Synthetic songs to analyse.")] = 100,
    seed: Annotated[int, typer.Option(help="Seed of the synthetic corpus.")] = 23,
) -> None:
    """Measure the harmonic analysis against synthetic songs with known harmony."""
    from backline.synth import generate_corpus
    from backline.theory.evaluate import evaluate

    for line in evaluate(generate_corpus(_load(config), n_songs=n, seed=seed)).lines():
        typer.echo(line)


if __name__ == "__main__":  # pragma: no cover
    app()
