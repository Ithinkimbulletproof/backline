# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Shared fixtures. Everything here is synthetic: no real tabs are ever used in tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from backline.config import BacklineConfig, load_config

if TYPE_CHECKING:
    from backline.synth import SynthSong

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def tiny_config() -> BacklineConfig:
    return load_config(CONFIGS / "tiny.yaml")


@pytest.fixture(scope="session")
def corpus(tiny_config: BacklineConfig) -> list[SynthSong]:
    from backline.synth import generate_corpus

    return generate_corpus(tiny_config, n_songs=24)


@pytest.fixture(scope="session")
def large_corpus(tiny_config: BacklineConfig) -> list[SynthSong]:
    from backline.synth import generate_corpus

    return generate_corpus(tiny_config, n_songs=120, seed=7)
