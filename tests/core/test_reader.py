# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

import pytest

from backline.formats.common import UnsupportedFormatError
from backline.formats.gpif_writer import write_gp
from backline.formats.guitarpro import write_gp5
from backline.formats.midi import write_midi
from backline.formats.reader import SUPPORTED_SUFFIXES, read_score
from backline.synth import SynthSong


@pytest.mark.parametrize(
    ("suffix", "writer"), [(".gp", write_gp), (".GP5", write_gp5), (".mid", write_midi)]
)
def test_dispatch_by_extension(
    corpus: list[SynthSong], tmp_path: Path, suffix: str, writer: object
) -> None:
    song = corpus[0]
    path = tmp_path / f"song{suffix}"
    writer(song.score, path)  # type: ignore[operator]
    score = read_score(path)
    assert score.n_bars == song.score.n_bars
    assert len(score.tracks) == 3


def test_gpx_is_reported_as_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "old.gpx"
    path.write_bytes(b"BCFZ")
    with pytest.raises(UnsupportedFormatError, match="Guitar Pro 6"):
        read_score(path)


def test_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("riff", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError, match=r"\.txt"):
        read_score(path)


def test_supported_suffixes() -> None:
    assert {".gp", ".gp3", ".gp4", ".gp5", ".mid", ".midi"} == SUPPORTED_SUFFIXES
