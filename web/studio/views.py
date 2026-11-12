# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

import backline


def index(request: HttpRequest) -> HttpResponse:
    """Studio start page."""
    return render(request, "studio/index.html", {"version": backline.__version__})
