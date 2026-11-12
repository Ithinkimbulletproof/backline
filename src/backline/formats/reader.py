# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""One entry point to read any supported file into a :class:`~backline.score.Score`."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from backline.formats.common import ReadStats, UnsupportedFormatError
from backline.formats.gpif import read_gp
from backline.formats.guitarpro import read_guitarpro
from backline.formats.midi import read_midi
from backline.score import Score

FORMAT_NAMES: Final = {
    ".gp": "Guitar Pro 7+ (GPIF)",
    ".gp5": "Guitar Pro 5",
    ".gp4": "Guitar Pro 4",
    ".gp3": "Guitar Pro 3",
    ".mid": "MIDI",
    ".midi": "MIDI",
}
SUPPORTED_SUFFIXES: Final = frozenset(FORMAT_NAMES)
KNOWN_UNSUPPORTED: Final = {".gpx": "Guitar Pro 6 (.gpx) files are not supported yet"}


def read_score(
    path: Path, *, stats: ReadStats | None = None, apply_triplet_feel: bool = True
) -> Score:
    """Read a tab or MIDI file, choosing the reader by extension.

    Raises :class:`~backline.formats.common.UnsupportedFormatError` for other extensions
    and :class:`~backline.formats.common.ScoreReadError` for files that cannot be parsed.
    """
    suffix = path.suffix.lower()
    if suffix in (".gp3", ".gp4", ".gp5"):
        return read_guitarpro(path, stats=stats, apply_triplet_feel=apply_triplet_feel)
    if suffix == ".gp":
        return read_gp(path, stats=stats, apply_triplet_feel=apply_triplet_feel)
    if suffix in (".mid", ".midi"):
        return read_midi(path)
    if suffix in KNOWN_UNSUPPORTED:
        raise UnsupportedFormatError(KNOWN_UNSUPPORTED[suffix])
    raise UnsupportedFormatError(f"unsupported file type: {suffix or '(no extension)'}")
