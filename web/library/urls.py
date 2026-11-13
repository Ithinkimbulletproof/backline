# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.urls import path

from web.library import views

app_name = "library"

urlpatterns = [
    path("", views.index, name="index"),
    path("export.csv", views.export, name="export"),
    path("model/", views.model_status, name="model"),
]
