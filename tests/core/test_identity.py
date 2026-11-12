# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import copy
from collections import Counter
from pathlib import Path

import pytest

from backline.ingest.identity import (
    SongKeys,
    assign_split,
    content_key,
    group_songs,
    name_key,
    normalize_name,
    resolve_metadata,
)
from backline.score import Score, Tempo, TimeSignature, TrackRole
from backline.synth import SynthSong


@pytest.mark.parametrize(
    ("raw", "artist", "expected"),
    [
        ("The Beatles", True, "beatles"),
        ("Кино", True, "kino"),
        ("Звезда по имени Солнце", False, "zvezda po imeni solntse"),
        ("Song (Live)", False, "song"),
        ("Song [v2]", False, "song"),
        ("Song ver. 2", False, "song"),
        ("Song 2", False, "song 2"),
        ("AC/DC", True, "ac dc"),
        ("Motörhead", True, "motorhead"),
        ("Artist feat. Somebody", True, "artist"),
        ("Rock & Roll", False, "rock and roll"),
        ("  THE   Who  ", True, "who"),
    ],
)
def test_normalize_name(raw: str, artist: bool, expected: str) -> None:
    assert normalize_name(raw, artist=artist) == expected


def _empty(title: str = "", artist: str = "") -> Score:
    return Score(1, [TimeSignature(0, 4, 4)], [Tempo(0, 120.0)], title=title, artist=artist)


def test_metadata_prefers_tags(tmp_path: Path) -> None:
    meta = resolve_metadata(
        _empty("Real Title", "Real Artist"),
        tmp_path / "x - y.gp",
        tmp_path,
        ["{artist} - {title}"],
        folder_is_artist=True,
    )
    assert (meta.artist, meta.title, meta.artist_source, meta.title_source) == (
        "Real Artist",
        "Real Title",
        "tags",
        "tags",
    )


def test_metadata_falls_back_to_file_and_folder(tmp_path: Path) -> None:
    patterns = ["{artist} - {title}"]
    from_name = resolve_metadata(
        _empty("Band - Song"),
        tmp_path / "Band - Song.gp",
        tmp_path,
        patterns,
        folder_is_artist=True,
    )
    assert (from_name.artist, from_name.title) == ("Band", "Song")
    assert (from_name.artist_source, from_name.title_source) == ("filename", "filename")

    from_folder = resolve_metadata(
        _empty(), tmp_path / "Band" / "Song.gp5", tmp_path, patterns, folder_is_artist=True
    )
    assert (from_folder.artist, from_folder.title, from_folder.artist_source) == (
        "Band",
        "Song",
        "folder",
    )
    unknown = resolve_metadata(
        _empty(), tmp_path / "Song.gp5", tmp_path, patterns, folder_is_artist=True
    )
    assert unknown.artist == ""
    assert name_key(unknown) is None


def test_content_key_ignores_transposition_names_and_track_order(corpus: list[SynthSong]) -> None:
    score = corpus[0].score
    moved = copy.deepcopy(score)
    for track in moved.tracks:
        track.name = "renamed"
        if not track.is_percussion:
            track.notes = [n.transposed(-3) for n in track.notes]
    moved.tracks.reverse()
    assert content_key(moved) == content_key(score)
    assert content_key(corpus[1].score) != content_key(score)


def test_content_key_sees_rhythm_changes(corpus: list[SynthSong]) -> None:
    score = corpus[0].score
    changed = copy.deepcopy(score)
    bass = next(t for t in changed.tracks if t.role is TrackRole.BASS)
    bass.notes = bass.notes[1:]
    assert content_key(changed) != content_key(score)


def test_groups_chain_through_names_and_content() -> None:
    songs = [
        SongKeys("1", "band|song", "c1"),
        SongKeys("2", "band|song", "c2"),
        SongKeys("3", None, "c2"),
        SongKeys("4", "other|tune", "c4"),
    ]
    groups = group_songs(songs)
    assert groups["1"] == groups["2"] == groups["3"]
    assert groups["4"] != groups["1"]
    assert group_songs(list(reversed(songs))) == groups


def test_split_is_deterministic_and_roughly_proportional() -> None:
    splits = [
        assign_split(f"group{i}", 7, val_fraction=0.1, test_fraction=0.05) for i in range(4000)
    ]
    again = [
        assign_split(f"group{i}", 7, val_fraction=0.1, test_fraction=0.05) for i in range(4000)
    ]
    assert splits == again
    counts = Counter(splits)
    assert abs(counts["val"] / 4000 - 0.1) < 0.02
    assert abs(counts["test"] / 4000 - 0.05) < 0.015
    assert counts["train"] > counts["val"] > counts["test"]
