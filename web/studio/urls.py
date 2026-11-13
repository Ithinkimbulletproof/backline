# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.urls import path

from web.studio import views

app_name = "studio"

urlpatterns = [
    path("", views.index, name="index"),
    path("adapters/", views.start_adapter, name="start-adapter"),
    path("jobs/<int:pk>/", views.job_status, name="job-status"),
]
