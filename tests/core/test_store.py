# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import gzip
from pathlib import Path

import pytest

from backline.formats.store import (
    ScoreStoreError,
    load_score,
    save_score,
    score_from_dict,
    score_to_dict,
)
from backline.synth import SynthSong


def test_round_trip(corpus: list[SynthSong], tmp_path: Path) -> None:
    score = corpus[0].score
    score.tracks[-1].percussion_names = {36: "Kick.Hit"}
    path = tmp_path / "song.json.gz"
    save_score(score, path)
    assert load_score(path) == score
    assert score_from_dict(score_to_dict(score)) == score


def test_unknown_version_is_rejected(corpus: list[SynthSong]) -> None:
    data = score_to_dict(corpus[0].score)
    data["version"] = 99
    with pytest.raises(ScoreStoreError, match="version"):
        score_from_dict(data)


def test_damaged_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("{not json")
    with pytest.raises(ScoreStoreError):
        load_score(path)
