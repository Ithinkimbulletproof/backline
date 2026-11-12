# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""``python manage.py build_originality_index``: index the library for copy detection."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from backline.config import ConfigError, PathSettings, load_config
from backline.ingest.pipeline import LibraryPaths
from backline.originality.build import build_originality_index


class Command(BaseCommand):
    help = "Build the originality index of the imported library and calibrate its thresholds."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--config", help="Profile YAML (default: BACKLINE_CONFIG)")

    def handle(self, *args: Any, **options: Any) -> None:
        paths = PathSettings()
        try:
            config = load_config(options["config"])
        except ConfigError as exc:
            raise CommandError(str(exc)) from exc
        library = LibraryPaths.from_settings(paths)
        if not library.catalog.is_file():
            raise CommandError("no catalog yet: run `python manage.py import_tabs` first")
        target = paths.artifacts_dir / "originality" / config.name
        index = build_originality_index(config, library, target)
        self.stdout.write(f"index: {target} ({len(index.songs)} songs, {index.n_groups} groups)")
        for part, limits in index.meta["thresholds"].items():
            self.stdout.write(
                f"{part}: yellow >= {limits['yellow']:.0%}, red >= {limits['red']:.0%}"
            )
