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


class Riff(models.Model):
    """An uploaded riff and everything chosen for it in the studio."""

    file = models.FileField(upload_to="uploads/")
    name = models.CharField(max_length=255)
    track = models.IntegerField(default=0)
    start_bar = models.PositiveIntegerField(default=0)
    n_bars = models.PositiveIntegerField(default=4)
    tonic = models.PositiveSmallIntegerField(default=0)
    mode = models.CharField(max_length=12, default="major")
    meter = models.CharField(max_length=8, default="4/4")
    tempo = models.FloatField(default=120.0)
    plan = models.JSONField(default=list)
    """Bars of beat cells holding chord symbols ("" continues the previous chord)."""
    references = models.JSONField(default=list)
    strength = models.FloatField(default=1.0)
    until_green = models.BooleanField(default=True)
    adapter_job = models.ForeignKey(AdapterJob, null=True, blank=True, on_delete=models.SET_NULL)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]

    def __str__(self) -> str:
        return self.name


class GenerationJob(models.Model):
    """A request to write a plan, or parts under the current plan."""

    class Kind(models.TextChoices):
        PLAN = "plan", "план"
        PARTS = "parts", "партии"

    riff = models.ForeignKey(Riff, related_name="jobs", on_delete=models.CASCADE)
    kind = models.CharField(max_length=5, choices=Kind.choices)
    status = models.CharField(
        max_length=8, choices=AdapterJob.Status.choices, default=AdapterJob.Status.QUEUED
    )
    message = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created"]

    @property
    def finished(self) -> bool:
        return self.status in (AdapterJob.Status.DONE, AdapterJob.Status.FAILED)


class Variant(models.Model):
    """One generated bass and drum part for a riff."""

    riff = models.ForeignKey(Riff, related_name="variants", on_delete=models.CASCADE)
    job = models.ForeignKey(GenerationJob, related_name="variants", on_delete=models.CASCADE)
    number = models.PositiveSmallIntegerField()
    plan = models.JSONField(default=list)
    verdict = models.CharField(max_length=8, blank=True)
    score = models.PositiveSmallIntegerField(default=100)
    originality = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    attempts = models.PositiveSmallIntegerField(default=1)
    warnings = models.JSONField(default=list)
    """Drum classes the output kit lacks and that this part plays with another sound."""
    roll = models.TextField(blank=True)
    """Pre-rendered SVG piano roll."""
    bass_file = models.CharField(max_length=255, blank=True)
    drums_file = models.CharField(max_length=255, blank=True)
    mix_file = models.CharField(max_length=255, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-job__created", "number"]
