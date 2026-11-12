# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Pronounceable made-up artist names and song titles for the synthetic corpus."""

from __future__ import annotations

import random
from typing import Final

_ONSETS: Final = ("b", "d", "g", "k", "l", "m", "n", "r", "s", "t", "v", "z", "br", "dr", "gr")
_MORE_ONSETS: Final = ("st", "tr", "sh", "kr", "f", "h", "p")
_VOWELS: Final = ("a", "e", "i", "o", "u", "ai", "ou", "ea")
_CODAS: Final = ("", "", "", "n", "r", "s", "l", "x", "th", "nd", "rk")


def _word(rng: random.Random, syllables: int) -> str:
    onsets = _ONSETS + _MORE_ONSETS
    parts = [rng.choice(onsets) + rng.choice(_VOWELS) for _ in range(syllables)]
    return ("".join(parts) + rng.choice(_CODAS)).capitalize()


def artist_name(rng: random.Random) -> str:
    """An invented band name, sometimes with a leading article."""
    words = " ".join(_word(rng, rng.randint(1, 3)) for _ in range(rng.randint(1, 2)))
    return f"The {words}" if rng.random() < 0.25 else words


def song_title(rng: random.Random) -> str:
    """An invented song title of one to three words."""
    return " ".join(_word(rng, rng.randint(1, 3)) for _ in range(rng.randint(1, 3)))
