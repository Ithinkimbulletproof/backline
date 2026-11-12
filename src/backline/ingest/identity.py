# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Who and what a song is: normalised names, content fingerprints, duplicate groups, split.

One song often exists as several tabs. Versions are grouped when their normalised
"artist + title" match or when their musical content matches up to transposition; a whole
group always lands in one split, so train and validation never share a song.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backline.score import Score, TrackRole

_TRANSLIT: Final = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
        "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
        "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
        "я": "ya", "і": "i", "ї": "yi", "є": "ye", "ґ": "g",
    }
)  # fmt: skip
_BRACKETS: Final = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_VERSION: Final = re.compile(r"\b(?:v|ver|version)\.?\s*\d+\b")
_FEATURING: Final = re.compile(r"\b(?:feat|ft|featuring)\b.*$")
_NON_WORD: Final = re.compile(r"[^a-z0-9]+")


def normalize_name(text: str, *, artist: bool = False) -> str:
    """Comparable form of an artist or title.

    Case, accents, Cyrillic vs Latin spelling, bracketed remarks ("(Live)", "[v2]"),
    version suffixes, "feat. ..." and punctuation are ignored; a leading "The" is dropped
    from artists.
    """
    value = unicodedata.normalize("NFKC", text).casefold()
    value = _BRACKETS.sub(" ", value)
    value = value.translate(_TRANSLIT)
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    value = value.replace("&", " and ")
    value = _FEATURING.sub(" ", value)
    value = _VERSION.sub(" ", value)
    value = _NON_WORD.sub(" ", value).strip()
    if artist:
        value = re.sub(r"^the\s+", "", value)
    return value


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    parts = re.split(r"(\{artist\}|\{title\})", pattern)
    regex = "".join(
        "(?P<artist>.+?)"
        if part == "{artist}"
        else "(?P<title>.+)"
        if part == "{title}"
        else re.escape(part)
        for part in parts
    )
    return re.compile(f"^{regex}$")


@dataclass(frozen=True, slots=True)
class Metadata:
    artist: str
    title: str
    artist_source: str
    """``tags``, ``filename``, ``folder`` or ``none``."""
    title_source: str
    """``tags`` or ``filename``."""


def resolve_metadata(
    score: Score,
    path: Path,
    root: Path,
    patterns: Sequence[str],
    *,
    folder_is_artist: bool,
) -> Metadata:
    """Artist and title from the tab's own fields, falling back to the file and folder name.

    Readers use the file stem as the title of untitled files, so a title equal to the stem
    counts as missing.
    """
    tag_title = score.title.strip()
    tag_title = "" if tag_title == path.stem else tag_title
    tag_artist = score.artist.strip()
    file_title = file_artist = ""
    for pattern in patterns:
        match = _pattern_regex(pattern).match(path.stem)
        if match is not None:
            file_title = match.group("title").strip()
            file_artist = (match.groupdict().get("artist") or "").strip()
            break

    title, title_source = (
        (tag_title, "tags") if tag_title else (file_title or path.stem, "filename")
    )
    if tag_artist:
        artist, artist_source = tag_artist, "tags"
    elif file_artist:
        artist, artist_source = file_artist, "filename"
    elif folder_is_artist and path.parent.resolve() != root.resolve():
        artist, artist_source = path.parent.name, "folder"
    else:
        artist, artist_source = "", "none"
    return Metadata(artist, title, artist_source, title_source)


def name_key(metadata: Metadata) -> str | None:
    """Grouping key from names, or ``None`` when the artist or title is unknown."""
    artist = normalize_name(metadata.artist, artist=True)
    title = normalize_name(metadata.title)
    if not artist or not title:
        return None
    return f"{artist}|{title}"


def content_key(score: Score) -> str:
    """Fingerprint of the musical content that ignores transposition and track names.

    Pitched tracks contribute (onset, pitch above the track's lowest note, duration) per
    note; drums contribute (onset, note). Track order within a role does not matter.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(
        repr([(ts.bar, ts.numerator, ts.denominator) for ts in score.time_signatures]).encode()
    )
    blocks: list[str] = []
    for track in score.tracks:
        if not track.notes or track.role is TrackRole.OTHER:
            continue
        rows: list[tuple[int, ...]]
        if track.is_percussion:
            rows = sorted((n.start, n.pitch) for n in track.notes)
        else:
            base = min(n.pitch for n in track.notes)
            rows = sorted((n.start, n.pitch - base, n.duration) for n in track.notes)
        blocks.append(f"{track.role.value}:{rows!r}")
    for block in sorted(blocks):
        digest.update(block.encode())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """SHA-1 of a file's bytes."""
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- groups


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass(frozen=True, slots=True)
class SongKeys:
    song_id: str
    name_key: str | None
    content_key: str


def group_songs(songs: Sequence[SongKeys]) -> dict[str, str]:
    """Map every song id to a group id; songs sharing a name key or content key group up.

    The group id is the smallest name key in the group (stable across re-imports), or the
    smallest content key when no version has usable names.
    """
    finder = _UnionFind(s.song_id for s in songs)
    first_by_key: dict[str, str] = {}
    for song in songs:
        for key in (
            f"name:{song.name_key}" if song.name_key else None,
            f"content:{song.content_key}",
        ):
            if key is None:
                continue
            if key in first_by_key:
                finder.union(first_by_key[key], song.song_id)
            else:
                first_by_key[key] = song.song_id
    members: dict[str, list[SongKeys]] = {}
    for song in songs:
        members.setdefault(finder.find(song.song_id), []).append(song)
    groups: dict[str, str] = {}
    for group in members.values():
        names = sorted(s.name_key for s in group if s.name_key)
        anchor = f"name:{names[0]}" if names else f"content:{min(s.content_key for s in group)}"
        group_id = hashlib.sha1(anchor.encode(), usedforsecurity=False).hexdigest()[:16]
        for song in group:
            groups[song.song_id] = group_id
    return groups


def assign_split(group_id: str, seed: int, *, val_fraction: float, test_fraction: float) -> str:
    """Deterministic split of a group: ``train``, ``val`` or ``test``."""
    digest = hashlib.sha1(f"{seed}:{group_id}".encode(), usedforsecurity=False).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < test_fraction:
        return "test"
    if position < test_fraction + val_fraction:
        return "val"
    return "train"
