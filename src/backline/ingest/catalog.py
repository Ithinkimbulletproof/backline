# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""The library catalog: one SQLite file with a row per tab file and per source track.

It belongs to the core package (standard-library ``sqlite3`` only); the web UI mirrors it
into its own models.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    file_hash TEXT NOT NULL,
    format TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    title_source TEXT NOT NULL DEFAULT '',
    artist_source TEXT NOT NULL DEFAULT '',
    name_key TEXT,
    content_key TEXT,
    group_id TEXT,
    canonical INTEGER NOT NULL DEFAULT 0,
    split TEXT,
    n_bars INTEGER NOT NULL DEFAULT 0,
    tempo REAL NOT NULL DEFAULT 0,
    meters TEXT NOT NULL DEFAULT '',
    n_guitars INTEGER NOT NULL DEFAULT 0,
    read_stats TEXT NOT NULL DEFAULT '{}',
    drum_stats TEXT NOT NULL DEFAULT '{}',
    imported_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS songs_group ON songs (group_id);
CREATE INDEX IF NOT EXISTS songs_status ON songs (status);
CREATE TABLE IF NOT EXISTS tracks (
    song_id TEXT NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    confidence REAL NOT NULL,
    ambiguous INTEGER NOT NULL,
    reasons TEXT NOT NULL,
    program INTEGER NOT NULL,
    instrument TEXT NOT NULL,
    tuning TEXT,
    n_notes INTEGER NOT NULL,
    pitch_min INTEGER,
    pitch_max INTEGER,
    stored_index INTEGER,
    PRIMARY KEY (song_id, idx)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

STATUSES: Final = ("ok", "duplicate", "filtered", "unsupported", "error")
"""``ok`` songs form the dataset; ``duplicate`` ones are other versions of an ``ok`` song."""


@dataclass(slots=True)
class TrackRecord:
    """A track of the source file with the role decided for it."""

    idx: int
    name: str
    role: str
    confidence: float
    ambiguous: bool
    reasons: list[str]
    program: int
    instrument: str
    tuning: list[int] | None
    n_notes: int
    pitch_min: int | None
    pitch_max: int | None
    stored_index: int | None
    """Position of the track in the stored song, or ``None`` if it was not kept."""


@dataclass(slots=True)
class SongRecord:
    """A tab file and what the import made of it."""

    id: str
    path: str
    size: int
    mtime: float
    file_hash: str
    format: str
    status: str
    reason: str = ""
    title: str = ""
    artist: str = ""
    title_source: str = ""
    artist_source: str = ""
    name_key: str | None = None
    content_key: str | None = None
    group_id: str | None = None
    canonical: bool = False
    split: str | None = None
    n_bars: int = 0
    tempo: float = 0.0
    meters: str = ""
    n_guitars: int = 0
    read_stats: dict[str, int] = field(default_factory=dict)
    drum_stats: dict[str, Any] = field(default_factory=dict)
    imported_at: str = ""
    tracks: list[TrackRecord] = field(default_factory=list)


_SONG_COLUMNS: Final = (
    "id", "path", "size", "mtime", "file_hash", "format", "status", "reason", "title",
    "artist", "title_source", "artist_source", "name_key", "content_key", "group_id",
    "canonical", "split", "n_bars", "tempo", "meters", "n_guitars", "read_stats",
    "drum_stats", "imported_at",
)  # fmt: skip
_TRACK_COLUMNS: Final = (
    "song_id", "idx", "name", "role", "confidence", "ambiguous", "reasons", "program",
    "instrument", "tuning", "n_notes", "pitch_min", "pitch_max", "stored_index",
)  # fmt: skip


class Catalog:
    """Read and write the catalog database."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._db:
            yield

    def songs(self, status: str | None = None) -> list[SongRecord]:
        """All songs, optionally only those with ``status``."""
        query = "SELECT * FROM songs" + (" WHERE status = ?" if status else "") + " ORDER BY path"
        rows = self._db.execute(query, (status,) if status else ()).fetchall()
        tracks: dict[str, list[TrackRecord]] = {}
        for row in self._db.execute("SELECT * FROM tracks ORDER BY song_id, idx"):
            tracks.setdefault(row["song_id"], []).append(_track_from_row(row))
        return [_song_from_row(row, tracks.get(row["id"], [])) for row in rows]

    def upsert(self, songs: Iterable[SongRecord]) -> None:
        """Insert or replace songs together with their tracks."""
        song_sql = (
            f"INSERT OR REPLACE INTO songs ({', '.join(_SONG_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _SONG_COLUMNS)})"
        )
        track_sql = (
            f"INSERT INTO tracks ({', '.join(_TRACK_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _TRACK_COLUMNS)})"
        )
        for song in songs:
            data = asdict(song)
            data["canonical"] = int(song.canonical)
            data["read_stats"] = json.dumps(song.read_stats, sort_keys=True)
            data["drum_stats"] = json.dumps(song.drum_stats, sort_keys=True)
            self._db.execute(song_sql, [data[c] for c in _SONG_COLUMNS])
            self._db.execute("DELETE FROM tracks WHERE song_id = ?", (song.id,))
            for track in song.tracks:
                row = asdict(track)
                row["song_id"] = song.id
                row["ambiguous"] = int(track.ambiguous)
                row["reasons"] = json.dumps(track.reasons, ensure_ascii=False)
                row["tuning"] = json.dumps(track.tuning) if track.tuning is not None else None
                self._db.execute(track_sql, [row[c] for c in _TRACK_COLUMNS])

    def remove(self, ids: Iterable[str]) -> None:
        for song_id in ids:
            self._db.execute("DELETE FROM songs WHERE id = ?", (song_id,))

    def set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])


def _song_from_row(row: sqlite3.Row, tracks: list[TrackRecord]) -> SongRecord:
    data = {c: row[c] for c in _SONG_COLUMNS}
    data["canonical"] = bool(data["canonical"])
    data["read_stats"] = json.loads(data["read_stats"])
    data["drum_stats"] = json.loads(data["drum_stats"])
    return SongRecord(**data, tracks=tracks)


def _track_from_row(row: sqlite3.Row) -> TrackRecord:
    data = {c: row[c] for c in _TRACK_COLUMNS if c != "song_id"}
    data["ambiguous"] = bool(data["ambiguous"])
    data["reasons"] = json.loads(data["reasons"])
    data["tuning"] = json.loads(data["tuning"]) if data["tuning"] is not None else None
    return TrackRecord(**data)
