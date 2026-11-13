# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only status of the pipeline: what is built, from what, and what to run next.

Everything here only reads files the core writes (catalog mirror, dataset and index
metadata, training log, adapter cache); nothing is started from the page.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from django.db.models import Count

from backline.adapters.cache import AdapterStore
from backline.config import BacklineConfig, PathSettings
from backline.dataset.build import dataset_dir
from backline.ingest.pipeline import LibraryPaths
from web.library.models import Song

SPARK_WIDTH, SPARK_HEIGHT, SPARK_POINTS = 320, 64, 160


@dataclass(slots=True)
class Step:
    """One stage of the pipeline as the page shows it."""

    title: str
    done: bool
    command: str
    facts: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class AdapterRow:
    key: str
    songs: str
    steps: int
    reference_share: float
    strength: float
    stopped_early: bool
    created: str


@dataclass(slots=True)
class PipelineStatus:
    config: str
    steps: list[Step]
    adapters: list[AdapterRow]
    loss_chart: str = ""
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)


def pipeline_status(config: BacklineConfig, paths: PathSettings) -> PipelineStatus:
    library = LibraryPaths.from_settings(paths)
    training, chart = _training(config, paths)
    index_step, thresholds = _index(config, paths)
    return PipelineStatus(
        config=config.name,
        steps=[_import(), _annotations(library), _dataset(config, paths), index_step, training],
        adapters=_adapters(paths),
        loss_chart=chart,
        thresholds=thresholds,
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _import() -> Step:
    counts = {r["status"]: r["n"] for r in Song.objects.values("status").annotate(n=Count("id"))}
    facts = [
        (label, str(counts[value])) for value, label in Song.Status.choices if counts.get(value)
    ]
    return Step(
        "Импорт табов",
        bool(counts.get(Song.Status.OK)),
        "python manage.py import_tabs",
        facts,
        "" if counts else "Положите табы в BACKLINE_TABS_DIR.",
    )


def _annotations(library: LibraryPaths) -> Step:
    ids = list(Song.objects.filter(status=Song.Status.OK).values_list("id", flat=True))
    done = sum(library.annotation_path(song_id).is_file() for song_id in ids)
    return Step(
        "Гармоническая разметка",
        bool(ids) and done == len(ids),
        "python manage.py annotate",
        [("размечено", f"{done} из {len(ids)}")] if ids else [],
    )


def _dataset(config: BacklineConfig, paths: PathSettings) -> Step:
    meta = _read_json(dataset_dir(paths.artifacts_dir, config.name) / "meta.json")
    report: dict[str, Any] = (meta or {}).get("report", {})
    facts = []
    if report:
        for split, label in (("train", "train"), ("val", "val"), ("test", "test")):
            songs = report.get("songs", {}).get(split, 0)
            segments = report.get("segments", {}).get(split, 0)
            tokens = report.get("tokens", {}).get(split, 0)
            if not songs:  # e.g. no test split configured
                continue
            facts.append(
                (label, f"{songs} песен · {segments} сегментов · {_thousands(tokens)} токенов")
            )
        lengths = report.get("length_percentiles", {})
        if lengths:
            facts.append(
                (
                    "длина p50/p90/max",
                    f"{lengths.get('p50')}/{lengths.get('p90')}/{lengths.get('max')}",
                )
            )
        facts.append(("словарь", str(report.get("vocab_size", "?"))))
    return Step("Датасет", meta is not None, "python manage.py build_dataset", facts)


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _index(config: BacklineConfig, paths: PathSettings) -> tuple[Step, dict[str, dict[str, float]]]:
    folder = paths.artifacts_dir / "originality" / config.name
    meta = _read_json(folder / "meta.json")
    thresholds: dict[str, dict[str, float]] = (meta or {}).get("thresholds", {})
    facts = []
    if meta is not None:
        songs = meta.get("songs")
        if isinstance(songs, list):
            facts.append(("песен в индексе", str(len(songs))))
        for part, label in (("bass", "бас"), ("drums", "барабаны")):
            if part in thresholds:
                value = thresholds[part]
                facts.append(
                    (
                        f"пороги ({label})",
                        f"жёлтый {value['yellow']:.0%} · красный {value['red']:.0%}",
                    )
                )
    ready = meta is not None and (folder / "index.npz").is_file()
    return Step(
        "Индекс оригинальности", ready, "python manage.py build_originality_index", facts
    ), thresholds


def _training(config: BacklineConfig, paths: PathSettings) -> tuple[Step, str]:
    folder = paths.checkpoints_dir / config.name
    records = _metrics(folder / "logs" / "metrics.jsonl")
    facts = []
    if records:
        last = records[-1]
        facts.append(("шаг", f"{int(last['step'])} из {config.train.max_steps}"))
        evaluated = [r for r in records if "selection_metric" in r]
        if evaluated:
            best = min(evaluated, key=lambda r: r["selection_metric"])
            facts.append(("лучший шаг", str(int(best["step"]))))
            for key, label, fmt in (
                ("val_loss", "val loss", "{:.3f}"),
                ("plan_adherence", "соблюдение плана", "{:.0%}"),
                ("bass_kick_coupling", "бас ↔ бочка", "{:.0%}"),
                ("validity", "валидность", "{:.0%}"),
                ("memorization", "доля красных", "{:.0%}"),
            ):
                value = best.get(key)
                if isinstance(value, float) and math.isfinite(value):
                    facts.append((label, fmt.format(value)))
    chart = sparkline([r["train_loss"] for r in records if "train_loss" in r])
    return (
        Step(
            "Базовая модель", (folder / "best.pt").is_file(), "python manage.py train_base", facts
        ),
        chart,
    )


def _metrics(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "step" in record:
            records.append(record)
    # A resumed run appends from its checkpoint step on: keep the latest record per step.
    by_step = {int(r["step"]): r for r in records}
    return [by_step[s] for s in sorted(by_step)]


def _adapters(paths: PathSettings) -> list[AdapterRow]:
    infos = AdapterStore(paths.adapters_dir).list()
    names = {
        s.id: str(s) for s in Song.objects.filter(id__in={r for i in infos for r in i.references})
    }
    return [
        AdapterRow(
            info.key,
            "; ".join(names.get(ref, ref) for ref in info.references),
            info.steps,
            info.reference_share,
            info.recommended_strength,
            info.stopped_early,
            info.created[:16].replace("T", " "),
        )
        for info in infos
    ]


def sparkline(values: list[float]) -> str:
    """A small SVG line of ``values`` on a log scale (only numbers go into the markup).

    A loss falls steeply at first and slowly afterwards; the log scale keeps the slow
    part visible.
    """
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return ""
    stride = max(len(finite) // SPARK_POINTS, 1)
    points = finite[::stride]
    heights = [math.log(v) for v in points] if min(points) > 0 else points
    lo, hi = min(heights), max(heights)
    span = hi - lo or 1.0
    dx = SPARK_WIDTH / (len(points) - 1)
    dy = (SPARK_HEIGHT - 8) / span
    coords = " ".join(
        f"{i * dx:.1f},{SPARK_HEIGHT - 4 - (v - lo) * dy:.1f}" for i, v in enumerate(heights)
    )
    return (
        f'<svg class="spark" viewBox="0 0 {SPARK_WIDTH} {SPARK_HEIGHT}" role="img" '
        f'aria-label="train loss: {points[0]:.3f} → {points[-1]:.3f}">'
        f'<polyline points="{coords}"/></svg>'
    )
