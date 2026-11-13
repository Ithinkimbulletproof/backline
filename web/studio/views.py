# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Studio pages and the HTMX partials they swap in."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

import backline
from backline.formats.common import ScoreReadError
from backline.formats.reader import SUPPORTED_SUFFIXES
from backline.music import Mode, pitch_class_name, prefers_flats
from backline.tokenizer.plan_table import PlanTableError, function_cells, table_to_plan
from web.library.models import Song
from web.studio import services
from web.studio.models import AdapterJob, GenerationJob, Riff, Variant
from web.studio.tasks import generate_task, train_adapter_task

MODES = [(m.value, label) for m, label in (
    (Mode.MAJOR, "мажор"),
    (Mode.MINOR, "минор"),
    (Mode.MIXOLYDIAN, "миксолидийский"),
    (Mode.DORIAN, "дорийский"),
)]  # fmt: skip


def index(request: HttpRequest) -> HttpResponse:
    """Upload a riff, or go back to a recent one."""
    return render(
        request,
        "studio/index.html",
        {
            "version": backline.__version__,
            "riffs": Riff.objects.all()[:10],
            "model_ready": services.model_ready(),
            "suffixes": ", ".join(sorted(SUPPORTED_SUFFIXES)),
        },
    )


@require_POST
def upload(request: HttpRequest) -> HttpResponse:
    file = request.FILES.get("riff")
    if file is None or Path(file.name or "").suffix.lower() not in SUPPORTED_SUFFIXES:
        return render(
            request,
            "studio/index.html",
            {
                "error": "Загрузите MIDI или Guitar Pro (.mid, .gp, .gp3–.gp5).",
                "riffs": Riff.objects.all()[:10],
            },
            status=400,
        )
    riff = Riff(file=file, name=Path(file.name or "riff").stem)
    riff.save()
    try:
        services.analyse(riff)
    except (ScoreReadError, ValueError) as exc:
        riff.file.delete(save=False)
        riff.delete()
        return render(
            request,
            "studio/index.html",
            {"error": f"Не удалось прочитать файл: {exc}", "riffs": Riff.objects.all()[:10]},
            status=400,
        )
    riff.save()
    return redirect("studio:riff", pk=riff.pk)


# ------------------------------------------------------------------ the riff page


def _plan_context(riff: Riff, errors: list[str] | None = None) -> dict[str, Any]:
    beats = len(riff.plan[0]) if riff.plan else 0
    plan_job = riff.jobs.filter(kind=GenerationJob.Kind.PLAN).first()
    empty = not any(cell for row in riff.plan for cell in row)
    return {
        "riff": riff,
        # The first plan is proposed as soon as the page opens.
        "auto_plan": empty and plan_job is None and services.model_ready(),
        "rows": function_cells(riff.plan, riff.tonic),
        "beats": range(1, beats + 1),
        "plan_errors": errors or [],
        "plan_job": plan_job,
        "parts_job": riff.jobs.filter(kind=GenerationJob.Kind.PARTS).first(),
    }


def _context(riff: Riff) -> dict[str, Any]:
    flats = prefers_flats(riff.tonic, Mode(riff.mode))
    return {
        "riff": riff,
        "model_ready": services.model_ready(),
        "tonics": [(pc, pitch_class_name(pc, flats=flats)) for pc in range(12)],
        "modes": MODES,
        "tracks": services.track_choices(riff),
        "max_bars": services.max_bars(),
        "selected": Song.objects.filter(id__in=riff.references),
        **_plan_context(riff),
        "variants": riff.variants.select_related("job")[:12],
    }


def riff_page(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    return render(request, "studio/riff.html", _context(riff))


@require_POST
def riff_settings(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    error = ""
    try:
        riff.tonic = int(request.POST.get("tonic", riff.tonic)) % 12
        riff.mode = Mode(request.POST.get("mode", riff.mode)).value
        riff.tempo = min(max(float(request.POST.get("tempo", riff.tempo)), 30.0), 300.0)
        riff.track = int(request.POST.get("track", riff.track))
        start = max(int(request.POST.get("start_bar", riff.start_bar + 1)) - 1, 0)
        n_bars = min(max(int(request.POST.get("n_bars", riff.n_bars)), 1), services.max_bars())
    except (TypeError, ValueError):
        error = "Проверьте значения: тональность, темп и такты должны быть числами."
    else:
        reshape = n_bars != riff.n_bars
        riff.start_bar, riff.n_bars = start, n_bars
        if reshape:
            beats = len(riff.plan[0]) if riff.plan else 4
            riff.plan = (riff.plan + [["" for _ in range(beats)] for _ in range(n_bars)])[:n_bars]
        riff.save()
    response = render(request, "studio/partials/settings.html", {**_context(riff), "error": error})
    response["HX-Trigger"] = "plan-updated"
    return response


# ------------------------------------------------------------------ references


@require_GET
def reference_search(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    query = request.GET.get("q", "").strip()
    songs = Song.objects.filter(status=Song.Status.OK)
    if query:
        songs = songs.filter(title__icontains=query) | songs.filter(artist__icontains=query)
    return render(
        request,
        "studio/partials/reference_results.html",
        {"riff": riff, "songs": songs.order_by("artist", "title")[:25]},
    )


@require_POST
def toggle_reference(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    song = request.POST.get("song", "")
    refs = set(riff.references)
    if song in refs:
        refs.discard(song)
    elif Song.objects.filter(id=song).exists():
        refs.add(song)
    riff.references = sorted(refs)
    riff.adapter_job = None
    riff.save()
    response = render(request, "studio/partials/selected.html", _context(riff))
    response["HX-Trigger"] = "references-changed"
    return response


@require_POST
def strength(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    if "strength" in request.POST:
        with contextlib.suppress(ValueError):
            riff.strength = min(max(float(request.POST["strength"]), 0.0), 1.0)
    if "until_green" in request.POST:
        # A hidden "off" comes before the checkbox, so the last value wins.
        riff.until_green = request.POST.get("until_green") == "on"
    riff.save(update_fields=["strength", "until_green"])
    return HttpResponse(f"{riff.strength:.1f}")


@require_POST
def prepare_adapter(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    if riff.references:
        job = AdapterJob.objects.create(references=list(riff.references))
        riff.adapter_job = job
        riff.save(update_fields=["adapter_job"])
        train_adapter_task(job.pk)
    return render(request, "studio/partials/adapter_status.html", {"riff": riff})


@require_GET
def adapter_status(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    return render(request, "studio/partials/adapter_status.html", {"riff": riff})


# ------------------------------------------------------------------ plan and generation


@require_POST
def save_plan(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    beats = len(riff.plan[0]) if riff.plan else 4
    riff.plan = [
        [request.POST.get(f"cell-{bar}-{beat}", "").strip() for beat in range(beats)]
        for bar in range(riff.n_bars)
    ]
    riff.save(update_fields=["plan"])
    return render(request, "studio/partials/plan.html", _plan_context(riff, _plan_errors(riff)))


def _plan_errors(riff: Riff) -> list[str]:
    try:
        table_to_plan(riff.plan, riff.tonic, Mode(riff.mode), services.bar_steps(riff))
    except PlanTableError as exc:
        return [str(exc)]
    return []


@require_GET
def plan_partial(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    return render(request, "studio/partials/plan.html", _plan_context(riff))


@require_POST
def generate(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    kind = request.POST.get("kind", GenerationJob.Kind.PARTS)
    if kind == GenerationJob.Kind.PARTS:
        errors = _plan_errors(riff)
        if errors:
            return render(request, "studio/partials/plan.html", _plan_context(riff, errors))
    if not services.model_ready():
        return render(
            request,
            "studio/partials/plan.html",
            _plan_context(riff, ["Модель ещё не обучена: запустите `backline train`."]),
        )
    job = GenerationJob.objects.create(riff=riff, kind=kind)
    generate_task(job.pk)
    # With an immediate queue the job is already done here: show its result right away.
    job.refresh_from_db()
    riff.refresh_from_db()
    response = render(request, "studio/partials/plan.html", _plan_context(riff))
    if job.finished and job.kind == GenerationJob.Kind.PARTS:
        response["HX-Trigger"] = "variants-updated"
    return response


@require_GET
def generation_status(request: HttpRequest, pk: int) -> HttpResponse:
    job = get_object_or_404(GenerationJob, pk=pk)
    response = render(request, "studio/partials/job_status.html", {"job": job})
    if job.finished:
        response["HX-Trigger"] = (
            "plan-updated" if job.kind == GenerationJob.Kind.PLAN else "variants-updated"
        )
    return response


@require_GET
def variants_partial(request: HttpRequest, pk: int) -> HttpResponse:
    riff = get_object_or_404(Riff, pk=pk)
    return render(
        request,
        "studio/partials/variants.html",
        {"riff": riff, "variants": riff.variants.select_related("job")[:12]},
    )


@require_GET
def variant_detail(request: HttpRequest, pk: int) -> HttpResponse:
    variant = get_object_or_404(Variant, pk=pk)
    return render(request, "studio/partials/variant.html", {"variant": variant})


# ------------------------------------------------------------------ JSON endpoints


@require_POST
def start_adapter(request: HttpRequest) -> JsonResponse:
    """Queue the adapter for the posted ``ref`` song ids (instant when it is cached)."""
    references = sorted(set(request.POST.getlist("ref")))
    if not references:
        return JsonResponse({"error": "Выберите хотя бы одну песню-референс."}, status=400)
    job = AdapterJob.objects.create(references=references)
    train_adapter_task(job.pk)
    return JsonResponse({"id": job.pk, "status": job.status}, status=202)


@require_GET
def job_status(request: HttpRequest, pk: int) -> JsonResponse:
    job = get_object_or_404(AdapterJob, pk=pk)
    return JsonResponse(
        {
            "id": job.pk,
            "status": job.status,
            "label": job.get_status_display(),
            "progress": round(job.progress, 3),
            "message": job.message,
            "key": job.key,
            "finished": job.finished,
        }
    )
