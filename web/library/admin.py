# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.contrib import admin

from web.library.models import Song, Track


class TrackInline(admin.TabularInline):
    model = Track
    extra = 0
    readonly_fields = [f.name for f in Track._meta.fields]
    can_delete = False


@admin.register(Song)
class SongAdmin(admin.ModelAdmin):
    list_display = ("artist", "title", "status", "split", "canonical", "n_bars", "n_guitars")
    list_filter = ("status", "split", "canonical")
    search_fields = ("artist", "title", "path")
    inlines = [TrackInline]
