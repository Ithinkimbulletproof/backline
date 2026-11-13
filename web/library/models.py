# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Web-side mirror of the core catalog, kept in sync by ``import_tabs``."""

from django.db import models


class Song(models.Model):
    """A tab file of the library and what the import made of it."""

    class Status(models.TextChoices):
        OK = "ok", "в датасете"
        DUPLICATE = "duplicate", "другая версия"
        FILTERED = "filtered", "отфильтрована"
        UNSUPPORTED = "unsupported", "формат не поддерживается"
        ERROR = "error", "ошибка чтения"

    class Category(models.TextChoices):
        """What it takes to use the file (see ``backline triage``)."""

        FIXABLE = "fixable", "можно поправить"
        REJECTED = "rejected", "не подходит"
        REVIEW = "review", "стоит проверить"
        OK = "ok", "годится"
        DUPLICATE = "duplicate", "другие версии"

    id = models.CharField(primary_key=True, max_length=16)
    path = models.CharField(max_length=1024)
    title = models.CharField(max_length=255, blank=True)
    artist = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices)
    reason = models.TextField(blank=True)
    category = models.CharField(
        max_length=16, choices=Category.choices, default=Category.OK, db_index=True
    )
    problems = models.JSONField(default=list)
    """Triage problems as dicts (code, track, track_name, value, limit, detail)."""
    split = models.CharField(max_length=8, blank=True)
    group_id = models.CharField(max_length=16, blank=True, db_index=True)
    canonical = models.BooleanField(default=False)
    n_bars = models.PositiveIntegerField(default=0)
    tempo = models.FloatField(default=0)
    meters = models.CharField(max_length=64, blank=True)
    n_guitars = models.PositiveSmallIntegerField(default=0)
    imported_at = models.CharField(max_length=32, blank=True)

    class Meta:
        ordering = ["artist", "title", "path"]

    def __str__(self) -> str:
        return f"{self.artist or '?'} — {self.title or self.path}"


class Track(models.Model):
    """A track of a song's source file with the role the import gave it."""

    song = models.ForeignKey(Song, related_name="tracks", on_delete=models.CASCADE)
    idx = models.PositiveSmallIntegerField()
    name = models.CharField(max_length=255)
    role = models.CharField(max_length=8)
    confidence = models.FloatField(default=0)
    ambiguous = models.BooleanField(default=False)
    program = models.PositiveSmallIntegerField(default=0)
    instrument = models.CharField(max_length=64, blank=True)
    tuning = models.CharField(max_length=64, blank=True)
    n_notes = models.PositiveIntegerField(default=0)
    stored_index = models.PositiveSmallIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["song", "idx"]
        constraints = [models.UniqueConstraint(fields=["song", "idx"], name="unique_track_index")]

    def __str__(self) -> str:
        return f"{self.song_id}#{self.idx} {self.name} ({self.role})"
