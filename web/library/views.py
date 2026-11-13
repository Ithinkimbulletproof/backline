# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.db.models import Count
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from backline.config import PathSettings, load_config
from web.library.models import Song
from web.library.status import pipeline_status


def index(request: HttpRequest) -> HttpResponse:
    """Library overview: import status counts and the songs in the dataset."""
    counts = {
        row["status"]: row["n"] for row in Song.objects.values("status").annotate(n=Count("id"))
    }
    statuses = [(label, counts.get(value, 0)) for value, label in Song.Status.choices]
    songs = Song.objects.filter(status=Song.Status.OK)[:200]
    return render(
        request,
        "library/index.html",
        {"statuses": statuses, "songs": songs, "total": sum(counts.values())},
    )


def model_status(request: HttpRequest) -> HttpResponse:
    """What is built so far (dataset, index, model, adapters) and what to run next."""
    return render(
        request, "library/model.html", {"status": pipeline_status(load_config(), PathSettings())}
    )
