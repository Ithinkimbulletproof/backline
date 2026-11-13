# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""The corpus index: every song's event sequences plus sorted n-gram postings.

For each channel (bass intervals, bass degrees, drum beats) the index stores the event
codes of every song back to back, the bar of every event, and for the seed and n-gram
lengths a sorted array of hashes with (song, position) postings. Document frequency
counts song *groups*, so several tabs of one song do not make its patterns look common.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

_BASE: Final = np.uint64(1_000_003)
CHANNELS: Final = ("bass", "degree", "drums")


def ngram_hashes(codes: npt.NDArray[np.int64], n: int) -> npt.NDArray[np.uint64]:
    """Polynomial hash of every window of ``n`` codes."""
    if len(codes) < n:
        return np.zeros(0, dtype=np.uint64)
    count = len(codes) - n + 1
    values = codes.astype(np.uint64)
    hashes = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            hashes = hashes * _BASE + values[k : k + count]
    return hashes


@dataclass(slots=True)
class SongEntry:
    """An indexed song and the group of its versions."""

    song_id: str
    group: int
    artist: str
    title: str


@dataclass(slots=True)
class Postings:
    """Sorted hashes of all windows of length ``n`` with where they occur."""

    n: int
    hashes: npt.NDArray[np.uint64]
    songs: npt.NDArray[np.uint32]
    positions: npt.NDArray[np.uint32]
    unique: npt.NDArray[np.uint64]
    groups: npt.NDArray[np.uint32]
    """Number of distinct song groups containing each unique hash."""

    def lookup(self, value: np.uint64) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        lo = int(np.searchsorted(self.hashes, value, side="left"))
        hi = int(np.searchsorted(self.hashes, value, side="right"))
        return self.songs[lo:hi], self.positions[lo:hi]

    def group_frequency(self, values: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
        """Groups containing each hash (0 when unseen)."""
        if len(self.unique) == 0 or len(values) == 0:
            return np.zeros(len(values), dtype=np.int64)
        index = np.clip(np.searchsorted(self.unique, values), 0, len(self.unique) - 1)
        found = self.unique[index] == values
        return np.where(found, self.groups[index], 0).astype(np.int64)


@dataclass(slots=True)
class Channel:
    """Event codes of all songs for one representation, plus postings."""

    codes: npt.NDArray[np.int64]
    bars: npt.NDArray[np.int32]
    offsets: npt.NDArray[np.int64]
    """``offsets[i]:offsets[i+1]`` are song ``i``'s events."""
    postings: dict[int, Postings] = field(default_factory=dict)

    def song_codes(self, song: int) -> npt.NDArray[np.int64]:
        return self.codes[self.offsets[song] : self.offsets[song + 1]]

    def song_bars(self, song: int) -> npt.NDArray[np.int32]:
        return self.bars[self.offsets[song] : self.offsets[song + 1]]


def build_channel(
    sequences: list[npt.NDArray[np.int64]],
    bars: list[npt.NDArray[np.int32]],
    song_groups: list[int],
    lengths: Iterable[int],
) -> Channel:
    """Concatenate per-song sequences and index windows of each length in ``lengths``."""
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in sequences])
    codes = np.concatenate(sequences) if sequences else np.zeros(0, dtype=np.int64)
    bar_array = np.concatenate(bars) if bars else np.zeros(0, dtype=np.int32)
    channel = Channel(codes.astype(np.int64), bar_array.astype(np.int32), offsets)
    for n in sorted(set(lengths)):
        all_hashes, all_songs, all_positions = [], [], []
        for song, sequence in enumerate(sequences):
            hashes = ngram_hashes(sequence, n)
            all_hashes.append(hashes)
            all_songs.append(np.full(len(hashes), song, dtype=np.uint32))
            all_positions.append(np.arange(len(hashes), dtype=np.uint32))
        hashes = np.concatenate(all_hashes) if all_hashes else np.zeros(0, np.uint64)
        songs = np.concatenate(all_songs) if all_songs else np.zeros(0, np.uint32)
        positions = np.concatenate(all_positions) if all_positions else np.zeros(0, np.uint32)
        order = np.argsort(hashes, kind="stable")
        hashes, songs, positions = hashes[order], songs[order], positions[order]
        groups = (
            np.asarray(song_groups, dtype=np.int64)[songs] if len(songs) else np.zeros(0, np.int64)
        )
        pairs = (
            np.unique(np.stack([hashes, groups.astype(np.uint64)], axis=1), axis=0)
            if len(hashes)
            else np.zeros((0, 2), np.uint64)
        )
        unique, counts = (
            np.unique(pairs[:, 0], return_counts=True)
            if len(pairs)
            else (np.zeros(0, np.uint64), np.zeros(0, np.int64))
        )
        channel.postings[n] = Postings(
            n, hashes, songs, positions, unique.astype(np.uint64), counts.astype(np.uint32)
        )
    return channel


@dataclass(slots=True)
class OriginalityIndex:
    """Everything needed to check a part against the corpus."""

    songs: list[SongEntry]
    channels: dict[str, Channel]
    beats_per_bar: list[int]
    """Felt beats per bar of each song (drum beats -> bars)."""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len({s.group for s in self.songs})

    def idf(self, group_frequency: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Rarity weight: log((G + 1) / (df + 1)), 0 for unseen patterns."""
        weights = np.log((self.n_groups + 1) / (group_frequency + 1))
        return np.where(group_frequency > 0, weights, 0.0)

    def max_idf(self) -> float:
        return math.log((self.n_groups + 1) / 2)

    # ----------------------------------------------------------------- storage

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, npt.NDArray[Any]] = {
            "beats_per_bar": np.asarray(self.beats_per_bar, np.int32)
        }
        for name, channel in self.channels.items():
            arrays[f"{name}.codes"] = channel.codes
            arrays[f"{name}.bars"] = channel.bars
            arrays[f"{name}.offsets"] = channel.offsets
            for n, postings in channel.postings.items():
                prefix = f"{name}.{n}"
                arrays[f"{prefix}.hashes"] = postings.hashes
                arrays[f"{prefix}.songs"] = postings.songs
                arrays[f"{prefix}.positions"] = postings.positions
                arrays[f"{prefix}.unique"] = postings.unique
                arrays[f"{prefix}.groups"] = postings.groups
        save_arrays: Any = np.savez
        save_arrays(folder / "index.npz", **arrays)
        meta = dict(self.meta)
        meta["songs"] = [[s.song_id, s.group, s.artist, s.title] for s in self.songs]
        meta["postings"] = {name: sorted(c.postings) for name, c in self.channels.items()}
        (folder / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: Path) -> OriginalityIndex:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        with np.load(folder / "index.npz") as data:
            arrays = {k: data[k] for k in data.files}
        channels = {}
        for name, lengths in meta.pop("postings").items():
            channel = Channel(
                arrays[f"{name}.codes"], arrays[f"{name}.bars"], arrays[f"{name}.offsets"]
            )
            for n in lengths:
                prefix = f"{name}.{n}"
                channel.postings[int(n)] = Postings(
                    int(n),
                    arrays[f"{prefix}.hashes"],
                    arrays[f"{prefix}.songs"],
                    arrays[f"{prefix}.positions"],
                    arrays[f"{prefix}.unique"],
                    arrays[f"{prefix}.groups"],
                )
            channels[name] = channel
        songs = [SongEntry(s, int(g), a, t) for s, g, a, t in meta.pop("songs")]
        return cls(songs, channels, arrays["beats_per_bar"].tolist(), meta)
