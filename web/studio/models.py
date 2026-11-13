# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""State of long-running jobs started from the UI."""

from django.db import models


class AdapterJob(models.Model):
    """Training (or cache lookup) of the adapter for a set of reference songs."""

    class Status(models.TextChoices):
        QUEUED = "queued", "в очереди"
        RUNNING = "running", "обучается"
        DONE = "done", "готов"
        CACHED = "cached", "в кэше"
        FAILED = "failed", "ошибка"

    references = models.JSONField(default=list)
    status = models.CharField(max_length=8, choices=Status.choices, default=Status.QUEUED)
    progress = models.FloatField(default=0.0)
    """From 0 to 1."""
    message = models.TextField(blank=True)
    key = models.CharField(max_length=32, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created"]

    def __str__(self) -> str:
        return f"adapter job {self.pk} ({self.status})"

    @property
    def finished(self) -> bool:
        return self.status in (self.Status.DONE, self.Status.CACHED, self.Status.FAILED)
