# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Mirror the core catalog (SQLite) into the web models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from django.db import transaction

from backline.ingest.catalog import Catalog
from backline.music import pitch_name
from web.library.models import Song, Track


@dataclass(frozen=True, slots=True)
class SyncResult:
    created: int
    updated: int
    deleted: int


def sync_catalog(catalog_path: Path) -> SyncResult:
    """Replace the web-side library with the catalog's content."""
    with Catalog(catalog_path) as catalog:
        records = catalog.songs()
    existing = set(Song.objects.values_list("id", flat=True))
    seen = {r.id for r in records}
    with transaction.atomic():
        Song.objects.exclude(id__in=seen).delete()
        for record in records:
            Song.objects.update_or_create(
                id=record.id,
                defaults={
                    "path": record.path,
                    "title": record.title,
                    "artist": record.artist,
                    "status": record.status,
                    "reason": record.reason,
                    "split": record.split or "",
                    "group_id": record.group_id or "",
                    "canonical": record.canonical,
                    "n_bars": record.n_bars,
                    "tempo": record.tempo,
                    "meters": record.meters,
                    "n_guitars": record.n_guitars,
                    "imported_at": record.imported_at,
                },
            )
        Track.objects.filter(song_id__in=seen).delete()
        Track.objects.bulk_create(
            Track(
                song_id=record.id,
                idx=track.idx,
                name=track.name[:255],
                role=track.role,
                confidence=track.confidence,
                ambiguous=track.ambiguous,
                program=track.program,
                instrument=track.instrument[:64],
                tuning=" ".join(pitch_name(p) for p in track.tuning)[:64] if track.tuning else "",
                n_notes=track.n_notes,
                stored_index=track.stored_index,
            )
            for record in records
            for track in record.tracks
        )
    created = len(seen - existing)
    return SyncResult(created=created, updated=len(seen) - created, deleted=len(existing - seen))
