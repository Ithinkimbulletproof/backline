# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic songs with known harmony, used by tests and CI instead of real tabs.

Every song comes with its ground truth (key per section, chord per span, role of every
bass note), so later stages can be tested against labels that are correct by construction.
"""

from backline.synth.song import (
    BassNoteTruth,
    ChordTruth,
    SectionTruth,
    SynthSong,
    generate_corpus,
    generate_song,
)

__all__ = [
    "BassNoteTruth",
    "ChordTruth",
    "SectionTruth",
    "SynthSong",
    "generate_corpus",
    "generate_song",
]
