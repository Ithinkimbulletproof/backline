# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Library pages: the imported tabs sorted by what it takes to use them."""

from __future__ import annotations

import csv
from typing import Any

from django.core.paginator import Paginator
from django.db.models import Case, Count, IntegerField, Q, QuerySet, Value, When
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from backline.config import PathSettings, load_config
from web.library.models import Song
from web.library.status import pipeline_status
from web.library.triage_text import describe

PAGE_SIZE = 100
ORDER = [value for value, _ in Song.Category.choices]
"""Categories in the order the page lists them: work to do first."""


def _filtered(request: HttpRequest) -> tuple[QuerySet[Song], str, str]:
    category = request.GET.get("category", "")
    query = request.GET.get("q", "").strip()
    songs = Song.objects.all()
    if category in ORDER:
        songs = songs.filter(category=category)
    else:
        category = ""
    if query:
        songs = songs.filter(
            Q(title__icontains=query) | Q(artist__icontains=query) | Q(path__icontains=query)
        )
    rank = Case(
        *(When(category=value, then=Value(i)) for i, value in enumerate(ORDER)),
        output_field=IntegerField(),
    )
    return songs.annotate(rank=rank).order_by("rank", "artist", "title", "path"), category, query


def index(request: HttpRequest) -> HttpResponse:
    """Every imported file with its category and what to fix, filterable and searchable."""
    counts = {
        row["category"]: row["n"] for row in Song.objects.values("category").annotate(n=Count("id"))
    }
    songs, category, query = _filtered(request)
    page = Paginator(songs, PAGE_SIZE).get_page(request.GET.get("page"))
    rows = [(song, [describe(p) for p in song.problems]) for song in page]
    return render(
        request,
        "library/index.html",
        {
            "categories": [
                (value, label, counts.get(value, 0)) for value, label in Song.Category.choices
            ],
            "category": category,
            "query": query,
            "page": page,
            "rows": rows,
            "total": sum(counts.values()),
            "in_dataset": Song.objects.filter(status=Song.Status.OK).count(),
        },
    )


def export(request: HttpRequest) -> HttpResponse:
    """The current selection as a CSV for Excel (semicolons, UTF-8 with a BOM)."""
    songs, _, _ = _filtered(request)
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="backline-tabs.csv"'
    response.write("\N{BYTE ORDER MARK}")  # lets Excel detect UTF-8
    writer = csv.writer(response, delimiter=";")
    writer.writerow(["Категория", "Файл", "Исполнитель", "Название", "Что не так и что сделать"])
    labels: dict[str, Any] = dict(Song.Category.choices)
    for song in songs:
        writer.writerow(
            [
                labels.get(song.category, song.category),
                song.path,
                song.artist,
                song.title,
                "; ".join(describe(p) for p in song.problems),
            ]
        )
    return response


def model_status(request: HttpRequest) -> HttpResponse:
    """What is built so far (dataset, index, model, adapters) and what to run next."""
    return render(
        request, "library/model.html", {"status": pipeline_status(load_config(), PathSettings())}
    )
