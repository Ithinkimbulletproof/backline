# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""``python manage.py annotate``: harmonic analysis of every imported song."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from backline.config import PathSettings
from backline.ingest.pipeline import LibraryPaths
from backline.theory.pipeline import annotate_library


class Command(BaseCommand):
    help = "Annotate imported songs: keys, chords per beat, functions, bass note roles."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--force", action="store_true", help="Re-annotate unchanged songs")

    def handle(self, *args: Any, **options: Any) -> None:
        library = LibraryPaths.from_settings(PathSettings())
        if not library.catalog.is_file():
            raise CommandError("no catalog yet: run `python manage.py import_tabs` first")
        for line in annotate_library(library, force=options["force"]).lines():
            self.stdout.write(line)
