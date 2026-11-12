# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""``python manage.py train_base``: train the base model on the built dataset."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from backline.config import ConfigError, PathSettings, load_config
from backline.dataset.build import dataset_dir
from backline.model.train import Trainer
from backline.originality.index import OriginalityIndex


class Command(BaseCommand):
    help = "Train (or resume training) the base model."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--config", help="Profile YAML (default: BACKLINE_CONFIG)")
        parser.add_argument("--dataset", help="Dataset name (default: profile name)")
        parser.add_argument("--device", help="cuda or cpu")
        parser.add_argument("--fresh", action="store_true", help="Start over")

    def handle(self, *args: Any, **options: Any) -> None:
        paths = PathSettings()
        try:
            config = load_config(options["config"])
        except ConfigError as exc:
            raise CommandError(str(exc)) from exc
        data = dataset_dir(paths.artifacts_dir, options["dataset"] or config.name)
        if not (data / "train.tokens").is_file():
            raise CommandError(f"no dataset at {data}: run `python manage.py build_dataset` first")
        index_dir = paths.artifacts_dir / "originality" / config.name
        index = OriginalityIndex.load(index_dir) if (index_dir / "index.npz").is_file() else None
        trainer = Trainer(
            config, data, paths.checkpoints_dir / config.name, device=options["device"], index=index
        )

        def progress(step: int, total: int, record: dict[str, float]) -> None:
            if "val_loss" in record:
                self.stdout.write(
                    f"step {step}/{total}: " + ", ".join(f"{k}={v:.4g}" for k, v in record.items())
                )

        summary = trainer.train(resume=not options["fresh"], progress=progress)
        self.stdout.write(f"best step {summary.best_step}, metric {summary.best_metric:.4f}")
