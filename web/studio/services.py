# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Studio logic between the views and the core: riffs, plans, generation, files."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from django.conf import settings

from backline.adapters.cache import AdapterStore
from backline.adapters.train import load_for_generation
from backline.config import BacklineConfig, PathSettings, load_config
from backline.drums import DrumMap
from backline.formats.midi import write_midi
from backline.formats.reader import read_score
from backline.model.metrics import WHOLE_NOTE_STEPS
from backline.model.service import Riff as RiffInput
from backline.model.service import generate_variants, prepare_riff
from backline.music import Mode, prefers_flats
from backline.originality.check import OriginalityChecker
from backline.originality.index import OriginalityIndex
from backline.score import Score, TrackRole
from backline.theory.grid import beats_per_bar
from backline.theory.key import KeySpan
from backline.tokenizer.plan_table import plan_to_table, table_to_plan
from backline.tokenizer.vocab import Vocab
from web.studio.models import AdapterJob, GenerationJob, Riff, Variant
from web.studio.render import piano_roll


@lru_cache(maxsize=1)
def config() -> BacklineConfig:
    return load_config()


def paths() -> PathSettings:
    return PathSettings()


def checkpoint_path() -> Path:
    return paths().checkpoints_dir / config().name / "best.pt"


def model_ready() -> bool:
    return checkpoint_path().is_file()


def max_bars() -> int:
    """The longest riff the model takes at once (its longest training segment)."""
    return max(config().tokenizer.segment_bars)


# ------------------------------------------------------------------ riffs


def riff_score(riff: Riff) -> Score:
    return read_score(Path(riff.file.path))


def track_choices(riff: Riff) -> list[tuple[int, str]]:
    return [(i, t.name or f"Трек {i + 1}") for i, t in enumerate(riff_score(riff).tracks)]


def analyse(riff: Riff) -> None:
    """Detect guitar track, bars, key, meter and tempo of a freshly uploaded riff."""
    prepared = prepare_riff(riff_score(riff), config())
    riff.track = prepared.guitar
    riff.start_bar = prepared.bars.start
    riff.n_bars = prepared.n_bars
    riff.tonic, riff.mode = prepared.key.tonic, prepared.key.mode.value
    meter = prepared.score.time_signature_at(prepared.bars.start)
    riff.meter = f"{meter.numerator}/{meter.denominator}"
    riff.tempo = round(prepared.score.tempo_at(prepared.score.bar_starts()[prepared.bars.start]), 1)
    riff.plan = [["" for _ in range(beats_per_bar(meter))] for _ in range(riff.n_bars)]


def riff_input(riff: Riff) -> RiffInput:
    """The core's view of a riff with the user's corrections applied."""
    score = riff_score(riff)
    score.tempos = [type(score.tempos[0])(0, riff.tempo)]
    bars = range(riff.start_bar, min(riff.start_bar + riff.n_bars, score.n_bars))
    key = KeySpan(bars.start, bars.stop, riff.tonic, Mode(riff.mode), True)
    prepared = prepare_riff(score, config(), bars=bars, track=riff.track)
    return RiffInput(prepared.score, prepared.guitar, bars, key)


def bar_steps(riff: Riff) -> int:
    numerator, denominator = (int(v) for v in riff.meter.split("/"))
    return numerator * WHOLE_NOTE_STEPS // denominator


# ------------------------------------------------------------------ model cache

_lock = threading.Lock()
_loaded: dict[str, Any] = {}


@dataclass(slots=True)
class Runtime:
    model: torch.nn.Module
    vocab: Vocab
    checker: OriginalityChecker | None
    adapter_key: str | None
    recommended_strength: float


def runtime(references: list[str]) -> Runtime:
    """The model (with the cached adapter for ``references`` if any), loaded once per process."""
    path = checkpoint_path()
    stamp = f"{path}:{path.stat().st_mtime_ns}:{','.join(sorted(references))}"
    with _lock:
        if _loaded.get("stamp") != stamp:
            model, info = load_for_generation(
                config(), path, references, AdapterStore(paths().adapters_dir)
            )
            index_dir = paths().artifacts_dir / "originality" / config().name
            checker = (
                OriginalityChecker(OriginalityIndex.load(index_dir), config())
                if (index_dir / "index.npz").is_file()
                else None
            )
            _loaded.clear()
            _loaded.update(
                stamp=stamp,
                runtime=Runtime(
                    model,
                    Vocab.from_config(config()),
                    checker,
                    info.key if info else None,
                    info.recommended_strength if info else 1.0,
                ),
            )
        result: Runtime = _loaded["runtime"]
        return result


# ------------------------------------------------------------------ generation


def output_folder(riff: Riff, job: GenerationJob) -> Path:
    return Path(settings.MEDIA_ROOT) / "generated" / str(riff.pk) / str(job.pk)


def run_generation(job_id: int) -> GenerationJob:
    """Write a plan (kind ``plan``) or variants under the current plan (kind ``parts``)."""
    job = GenerationJob.objects.select_related("riff").get(pk=job_id)
    job.status = AdapterJob.Status.RUNNING
    job.save()
    riff = job.riff
    try:
        _generate(job, riff)
    except Exception as exc:  # the UI shows the reason instead of a stuck job
        job.status, job.message = AdapterJob.Status.FAILED, str(exc)
    else:
        job.status, job.message = AdapterJob.Status.DONE, ""
    job.save()
    return job


def _generate(job: GenerationJob, riff: Riff) -> None:
    cfg = config()
    adapter_ready = riff.adapter_job is not None and riff.adapter_job.status in (
        AdapterJob.Status.DONE,
        AdapterJob.Status.CACHED,
    )
    references = list(riff.references) if adapter_ready else []
    rt = runtime(references)
    prepared = riff_input(riff)
    steps = bar_steps(riff)
    beats = len(riff.plan[0]) if riff.plan else 4
    flats = prefers_flats(riff.tonic, Mode(riff.mode))
    if job.kind == GenerationJob.Kind.PLAN:
        result = generate_variants(
            rt.model, rt.vocab, cfg, prepared, variants=1, strength=riff.strength, seed=job.pk
        )
        riff.plan = plan_to_table(
            result.variants[0].decoded.plan, prepared.n_bars, beats, steps, flats=flats
        )
        riff.save(update_fields=["plan"])
        return
    plan = table_to_plan(riff.plan, riff.tonic, Mode(riff.mode), steps)
    drum_map = cfg.drums.load_output_map()
    result = generate_variants(
        rt.model,
        rt.vocab,
        cfg,
        prepared,
        plan=plan,
        checker=rt.checker,
        references=list(riff.references),
        until_green=riff.until_green,
        seed=job.pk,
        drum_map=drum_map,
        strength=riff.strength if rt.adapter_key else 1.0,
    )
    folder = output_folder(riff, job)
    folder.mkdir(parents=True, exist_ok=True)
    groups = {note: cfg.drums.coarse_of(name) for note, name in cfg.drums.input_map.items()}
    groups.update(_output_groups(cfg))
    riff_track = prepared.score.tracks[prepared.guitar]
    for number, variant in enumerate(result.variants, start=1):
        files = {}
        for track in variant.score.tracks:
            single = Score(
                variant.score.n_bars, variant.score.time_signatures, variant.score.tempos, [track]
            )
            name = f"variant_{number}_{track.role.value}.mid"
            write_midi(single, folder / name)
            files[track.role] = name
        mix = Score(
            variant.score.n_bars,
            variant.score.time_signatures,
            variant.score.tempos,
            [_shifted_riff(riff_track, prepared), *variant.score.tracks],
        )
        write_midi(mix, folder / f"variant_{number}_mix.mid")
        report = variant.originality
        matched = (
            sorted({bar for p in report.parts for bar in p.corpus.matched_bars}) if report else []
        )
        relative = f"generated/{riff.pk}/{job.pk}/"
        Variant.objects.create(
            riff=riff,
            job=job,
            number=number,
            plan=plan_to_table(variant.decoded.plan, prepared.n_bars, beats, steps, flats=flats),
            verdict=report.verdict if report else "",
            score=report.score if report else 100,
            originality=_report_dict(report),
            metrics={
                "plan_adherence": variant.metrics.plan_adherence,
                "bass_kick_coupling": variant.metrics.bass_kick_coupling,
                "valid": variant.metrics.valid,
            },
            attempts=variant.attempts,
            warnings=substitution_warnings(
                drum_map, {hit.drum_class for hit in variant.decoded.drums}
            ),
            roll=piano_roll(variant.score, groups, matched),
            bass_file=relative + files[TrackRole.BASS],
            drums_file=relative + files[TrackRole.DRUMS],
            mix_file=relative + f"variant_{number}_mix.mid",
        )


DRUM_LABELS = {
    "kick": "Kick",
    "snare": "Snare",
    "rim": "Side Stick",
    "hh_closed": "Hi-Hat Closed",
    "hh_pedal": "Hi-Hat Pedal",
    "hh_open": "Hi-Hat Open",
    "tom_floor": "Floor Tom",
    "tom_mid": "Mid Tom",
    "tom_high": "High Tom",
    "crash1": "Crash 1",
    "crash2": "Crash 2",
    "ride": "Ride",
    "ride_bell": "Ride Bell",
    "china": "China",
    "splash": "Splash",
}


def substitution_warnings(drum_map: DrumMap, used: set[str]) -> list[str]:
    """A warning for each class the kit lacks that the part plays with another sound."""
    warnings = []
    for item in drum_map.substitutions():
        if item.missing in used:
            sound = DRUM_LABELS.get(item.played_as or "", f"нота {item.note}")
            warnings.append(f"{item.missing} в ките нет — сыграно {sound}")
    return warnings


def _output_groups(cfg: BacklineConfig) -> dict[int, str]:
    drum_map = cfg.drums.load_output_map()
    return {entry.note: cfg.drums.coarse_of(name) for name, entry in drum_map.classes.items()}


def _shifted_riff(track: Any, prepared: RiffInput) -> Any:
    from dataclasses import replace

    starts = prepared.score.bar_starts()
    lo, hi = starts[prepared.bars.start], starts[prepared.bars.stop]
    notes = [replace(n, start=n.start - lo) for n in track.notes if lo <= n.start < hi]
    return replace(track, notes=notes, role=TrackRole.GUITAR)


def _report_dict(report: Any) -> dict[str, Any]:
    if report is None:
        return {}
    parts = []
    for part in report.parts:
        sources = [
            {
                "artist": s.artist,
                "title": s.title,
                "bars": s.bars,
                "source_bars": list(s.source_bars),
            }
            for s in part.corpus.sources[:3]
        ]
        parts.append(
            {
                "part": part.part,
                "verdict": part.verdict,
                "score": part.score,
                "longest_bars": part.corpus.longest_bars,
                "rare_share": part.corpus.rare_share,
                "reference_longest": part.references.longest_bars if part.references else None,
                "sources": sources,
                "reasons": part.reasons,
            }
        )
    return {"verdict": report.verdict, "score": report.score, "parts": parts}
