# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""``python manage.py import_tabs [folder]``: run the core import and mirror the catalog."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from backline.config import ConfigError, PathSettings, load_config
from backline.ingest.pipeline import LibraryPaths, run_ingest
from web.library.sync import sync_catalog


class Command(BaseCommand):
    help = "Import a tab library (Guitar Pro / MIDI) and update the web library."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("source", nargs="?", help="Library folder (default: BACKLINE_TABS_DIR)")
        parser.add_argument("--config", help="Profile YAML (default: BACKLINE_CONFIG)")
        parser.add_argument("--workers", type=int, default=None, help="Reader processes")
        parser.add_argument("--force", action="store_true", help="Re-read unchanged files")

    def handle(self, *args: Any, **options: Any) -> None:
        paths = PathSettings()
        try:
            config = load_config(options["config"])
        except ConfigError as exc:
            raise CommandError(str(exc)) from exc
        library = LibraryPaths.from_settings(paths)
        source = Path(options["source"]) if options["source"] else paths.tabs_dir
        try:
            report = run_ingest(
                source, config, library, workers=options["workers"], force=options["force"]
            )
        except FileNotFoundError as exc:
            raise CommandError(str(exc)) from exc
        for line in report.lines():
            self.stdout.write(line)
        result = sync_catalog(library.catalog)
        self.stdout.write(
            self.style.SUCCESS(
                f"web library: {result.created} new, {result.updated} updated, "
                f"{result.deleted} removed"
            )
        )
