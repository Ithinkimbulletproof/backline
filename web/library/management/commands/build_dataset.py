# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""``python manage.py build_dataset``: tokenise the imported library."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from backline.config import ConfigError, PathSettings, load_config
from backline.dataset.build import build_dataset, dataset_dir
from backline.ingest.pipeline import LibraryPaths


class Command(BaseCommand):
    help = "Tokenise imported, annotated songs into train/val/test shards."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--config", help="Profile YAML (default: BACKLINE_CONFIG)")
        parser.add_argument("--name", help="Dataset folder name (default: profile name)")

    def handle(self, *args: Any, **options: Any) -> None:
        paths = PathSettings()
        try:
            config = load_config(options["config"])
        except ConfigError as exc:
            raise CommandError(str(exc)) from exc
        library = LibraryPaths.from_settings(paths)
        if not library.catalog.is_file():
            raise CommandError("no catalog yet: run `python manage.py import_tabs` first")
        target = dataset_dir(paths.artifacts_dir, options["name"] or config.name)
        for line in build_dataset(config, library, target).lines():
            self.stdout.write(line)
