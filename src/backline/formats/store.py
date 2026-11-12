# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Compact, versioned JSON storage of a :class:`~backline.score.Score` (``.json.gz``).

Imported songs are stored in this form so later stages never re-parse tab files. Notes are
written as ``[start, pitch, duration, velocity, flags]`` rows to keep files small.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Final

from backline.score import Note, NoteFlag, Score, Tempo, TimeSignature, Track, TrackRole

FORMAT_VERSION: Final = 1


class ScoreStoreError(ValueError):
    """Raised when stored data does not describe a valid score."""


def score_to_dict(score: Score) -> dict[str, Any]:
    """Plain JSON-serialisable form of a score."""
    return {
        "version": FORMAT_VERSION,
        "tpq": score.tpq,
        "title": score.title,
        "artist": score.artist,
        "n_bars": score.n_bars,
        "time_signatures": [[ts.bar, ts.numerator, ts.denominator] for ts in score.time_signatures],
        "tempos": [[t.tick, t.bpm] for t in score.tempos],
        "tracks": [
            {
                "name": track.name,
                "role": track.role.value,
                "program": track.program,
                "is_percussion": track.is_percussion,
                "tuning": list(track.tuning) if track.tuning is not None else None,
                "instrument": track.instrument,
                "percussion_names": {str(k): v for k, v in track.percussion_names.items()},
                "notes": [
                    [n.start, n.pitch, n.duration, n.velocity, int(n.flags)] for n in track.notes
                ],
            }
            for track in score.tracks
        ],
    }


def score_from_dict(data: dict[str, Any]) -> Score:
    """Rebuild a score from :func:`score_to_dict` output."""
    if data.get("version") != FORMAT_VERSION:
        raise ScoreStoreError(f"unsupported score format version: {data.get('version')!r}")
    try:
        tracks = [
            Track(
                name=t["name"],
                role=TrackRole(t["role"]),
                notes=[Note(s, p, d, v, NoteFlag(f)) for s, p, d, v, f in t["notes"]],
                program=t["program"],
                is_percussion=t["is_percussion"],
                tuning=tuple(t["tuning"]) if t["tuning"] is not None else None,
                instrument=t.get("instrument", ""),
                percussion_names={int(k): v for k, v in t.get("percussion_names", {}).items()},
            )
            for t in data["tracks"]
        ]
        return Score(
            n_bars=data["n_bars"],
            time_signatures=[TimeSignature(*row) for row in data["time_signatures"]],
            tempos=[Tempo(tick, bpm) for tick, bpm in data["tempos"]],
            tracks=tracks,
            tpq=data["tpq"],
            title=data.get("title", ""),
            artist=data.get("artist", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreStoreError(f"invalid stored score: {exc}") from exc


def save_score(score: Score, path: Path) -> None:
    """Write a score as gzip-compressed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(score_to_dict(score), ensure_ascii=False, separators=(",", ":"))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)


def load_score(path: Path) -> Score:
    """Read a score written by :func:`save_score`."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreStoreError(f"cannot read stored score {path}: {exc}") from exc
    return score_from_dict(data)
