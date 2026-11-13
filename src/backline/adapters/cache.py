# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Adapter cache: one folder per (references, settings, base model) combination.

The key hashes the sorted reference ids, the adapter settings and a fingerprint of the base
checkpoint, so choosing the same references again is instant, and retraining the base
model never serves a stale adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backline.config import AdapterConfig

META = "meta.json"


def file_fingerprint(path: Path) -> str:
    """SHA-256 of a file (the base checkpoint), memoised next to it by size and mtime."""
    stat = path.stat()
    memo = path.with_suffix(path.suffix + ".sha256")
    stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    if memo.is_file():
        saved_stamp, _, digest = memo.read_text(encoding="utf-8").partition(" ")
        if saved_stamp == stamp:
            return digest.strip()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    memo.write_text(f"{stamp} {digest}", encoding="utf-8")
    return digest


def adapter_key(references: Sequence[str], config: AdapterConfig, base: str) -> str:
    """Cache key of an adapter."""
    payload = json.dumps(
        {
            "references": sorted(set(references)),
            "adapter": config.model_dump(mode="json"),
            "base": base,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


@dataclass(slots=True)
class AdapterInfo:
    """What a cached adapter is and how it went."""

    key: str
    references: list[str]
    base: str
    steps: int = 0
    reference_share: float = 0.0
    stopped_early: bool = False
    recommended_strength: float = 1.0
    history: list[dict[str, float]] = field(default_factory=list)
    strength_check: dict[str, float] = field(default_factory=dict)
    """Share of red reference prompts at each tried strength."""
    created: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> AdapterInfo:
        data: dict[str, Any] = json.loads(text)
        return cls(**data)


class AdapterStore:
    """Folders of cached adapters under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        folder = self.path(key)
        return (folder / META).is_file() and (folder / "adapter_config.json").is_file()

    def info(self, key: str) -> AdapterInfo:
        return AdapterInfo.from_json((self.path(key) / META).read_text(encoding="utf-8"))

    def save_info(self, info: AdapterInfo) -> None:
        folder = self.path(info.key)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / META).write_text(info.to_json(), encoding="utf-8")

    def list(self) -> list[AdapterInfo]:
        if not self.root.is_dir():
            return []
        return sorted(
            (self.info(p.name) for p in self.root.iterdir() if self.exists(p.name)),
            key=lambda info: info.created,
            reverse=True,
        )
