# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""A small synthetic tab library with every situation the import has to handle."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from backline.config import BacklineConfig
from backline.formats.gpif_writer import write_gp
from backline.formats.guitarpro import write_gp5
from backline.formats.midi import write_midi
from backline.score import Note, Score, TrackRole
from backline.synth import generate_corpus


def _transposed(score: Score, semitones: int) -> Score:
    result = copy.deepcopy(score)
    for track in result.tracks:
        if not track.is_percussion:
            track.notes = [n.transposed(semitones) for n in track.notes]
            if track.tuning:
                track.tuning = tuple(p + semitones for p in track.tuning)
    return result


def _named(score: Score, *, artist: str | None = None, title: str | None = None) -> Score:
    result = copy.deepcopy(score)
    if artist is not None:
        result.artist = artist
    if title is not None:
        result.title = title
    return result


def build_library(root: Path, config: BacklineConfig) -> dict[str, str]:
    """Write the library under ``root``; returns label -> path relative to ``root``."""
    songs = [s.score for s in generate_corpus(config, n_songs=12, seed=4242)]
    with_lead = next(s for s in songs if len(s.tracks_with_role(TrackRole.GUITAR)) == 2)
    plain = [s for s in songs if s is not with_lead]
    paths: dict[str, str] = {}

    def put(label: str, relative: str, score: Score, writer: object) -> None:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        writer(score, target)  # type: ignore[operator]
        paths[label] = relative

    a = _named(plain[0], artist="Iron Lung", title="Static Field")
    put("a", "Iron Lung - Static Field.gp", a, write_gp)
    put("a_copy", "copies/Iron Lung - Static Field (copy).gp5", a, write_gp5)

    b = _named(plain[1], artist="Grey Harbor", title="Low Tide")
    put("b", "Grey Harbor/Low Tide.gp5", b, write_gp5)
    put(
        "b_live",
        "live/Low Tide live.gp",
        _named(_transposed(b, 2), title="Low Tide (Live)"),
        write_gp,
    )

    c = _named(plain[2], artist="The Velvet Engines", title="Night Drive")
    put("c", "Night Drive.gp", c, write_gp)
    put(
        "c_other",
        "Night Drive v2.gp5",
        _named(plain[3], artist="Velvet Engines", title="Night Drive"),
        write_gp5,
    )

    no_bass = _named(plain[4], artist="Solo Act", title="Bassless")
    no_bass.tracks = [t for t in no_bass.tracks if t.role is not TrackRole.BASS]
    put("no_bass", "Solo Act - Bassless.gp5", no_bass, write_gp5)

    intro_only = _named(plain[4], artist="Half Done", title="Intro Bass")
    bass_track = next(t for t in intro_only.tracks if t.role is TrackRole.BASS)
    two_bars = intro_only.bar_starts()[2]
    bass_track.notes = [n for n in bass_track.notes if n.start < two_bars]
    put("bass_intro", "Half Done - Intro Bass.gp5", intro_only, write_gp5)

    cowbell = _named(plain[5], artist="Copper Bell", title="More Cowbell")
    drums = next(t for t in cowbell.tracks if t.is_percussion)
    drums.notes = sorted([*drums.notes, *(Note(i * 1920, 56, 240) for i in range(cowbell.n_bars))])
    put("cowbell", "Copper Bell - More Cowbell.gp", cowbell, write_gp)

    odd = _named(plain[6], artist="Odd Signal", title="Mislabelled")
    bass_index = next(i for i, t in enumerate(odd.tracks) if t.role is TrackRole.BASS)
    odd.tracks[bass_index] = replace(odd.tracks[bass_index], name="Track 3", program=30)
    put("ambiguous", "Odd Signal - Mislabelled.gp5", odd, write_gp5)

    kino = _named(plain[7], artist="Кино", title="Звезда")
    put("cyrillic", "Кино - Звезда.gp", kino, write_gp)
    put("latin", "Kino - Zvezda.mid", _named(plain[7], title="Zvezda"), write_midi)

    lead_first = _named(with_lead, artist="Twin Axe", title="Harmony")
    rhythm, lead = lead_first.tracks_with_role(TrackRole.GUITAR)
    others = [t for t in lead_first.tracks if t.role is not TrackRole.GUITAR]
    lead_first.tracks = [lead, rhythm, *others]
    put("lead_first", "Twin Axe - Harmony.gp", lead_first, write_gp)
    paths["rhythm_name"] = rhythm.name

    for label, index in (("solo_1", 8), ("solo_2", 9), ("solo_3", 10)):
        score = plain[index]
        put(label, f"{score.artist} - {score.title}.gp5", score, write_gp5)

    (root / "old").mkdir(exist_ok=True)
    (root / "old" / "legacy.gpx").write_bytes(b"BCFZ\x00\x00")
    paths["gpx"] = "old/legacy.gpx"
    (root / "broken.gp").write_bytes(b"this is not a zip")
    paths["broken"] = "broken.gp"
    (root / "notes.txt").write_text("not a tab", encoding="utf-8")
    return paths
