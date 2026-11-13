# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.urls import path

from web.studio import views

app_name = "studio"

urlpatterns = [
    path("", views.index, name="index"),
    path("riffs/", views.upload, name="upload"),
    path("riffs/<int:pk>/", views.riff_page, name="riff"),
    path("riffs/<int:pk>/settings/", views.riff_settings, name="settings"),
    path("riffs/<int:pk>/references/", views.reference_search, name="reference-search"),
    path("riffs/<int:pk>/references/toggle/", views.toggle_reference, name="toggle-reference"),
    path("riffs/<int:pk>/strength/", views.strength, name="strength"),
    path("riffs/<int:pk>/adapter/", views.prepare_adapter, name="prepare-adapter"),
    path("riffs/<int:pk>/adapter/status/", views.adapter_status, name="adapter-status"),
    path("riffs/<int:pk>/plan/", views.plan_partial, name="plan"),
    path("riffs/<int:pk>/plan/save/", views.save_plan, name="save-plan"),
    path("riffs/<int:pk>/generate/", views.generate, name="generate"),
    path("riffs/<int:pk>/variants/", views.variants_partial, name="variants"),
    path("generation/<int:pk>/", views.generation_status, name="generation-status"),
    path("variants/<int:pk>/", views.variant_detail, name="variant"),
    path("adapters/", views.start_adapter, name="start-adapter"),
    path("jobs/<int:pk>/", views.job_status, name="job-status"),
]
