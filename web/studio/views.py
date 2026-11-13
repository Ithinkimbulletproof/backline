# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET, require_POST

import backline
from web.studio.models import AdapterJob
from web.studio.tasks import train_adapter_task


def index(request: HttpRequest) -> HttpResponse:
    """Studio start page."""
    return render(request, "studio/index.html", {"version": backline.__version__})


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
