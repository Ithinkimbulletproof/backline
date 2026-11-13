# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""URL routes of the web UI."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("library/", include("web.library.urls")),
    path("", include("web.studio.urls")),
]

# Generated MIDI files are served by the dev server (the UI is local-only).
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
