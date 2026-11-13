# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a tokenised dataset from the imported, annotated library.

Layout of ``<artifacts>/datasets/<name>/``::

    vocab.json        token list and fingerprint
    meta.json         config name, counts, length statistics
    songs.json        song ids, indexed by the ``song`` field of the index
    <split>.tokens    uint16 token ids of all segments, back to back
    <split>.index.npy one record per segment (see INDEX_DTYPE)

Segments longer than the model context are dropped and counted.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from backline.config import BacklineConfig
from backline.formats.store import load_score
from backline.ingest.catalog import Catalog
from backline.ingest.pipeline import LibraryPaths
from backline.theory.annotate import annotate_score, load_annotation, save_annotation
from backline.tokenizer.encode import EncodeError, SegmentEncoder
from backline.tokenizer.segments import song_segments
from backline.tokenizer.vocab import Vocab

INDEX_DTYPE: Final = np.dtype(
    [
        ("offset", "<u8"),
        ("length", "<u4"),
        ("target", "<u4"),
        ("song", "<u4"),
        ("start_bar", "<u2"),
        ("n_bars", "<u1"),
        ("guitar_lo", "<u1"),
        ("guitar_hi", "<u1"),
        ("bass_lo", "<u1"),
        ("bass_hi", "<u1"),
    ]
)
SPLITS: Final = ("train", "val", "test")

ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class DatasetReport:
    """Counts per split, dropped segments and length statistics of a built dataset."""

    name: str
    path: str
    songs: dict[str, int]
    segments: dict[str, int]
    tokens: dict[str, int]
    dropped: dict[str, int]
    length_percentiles: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0

    def lines(self) -> list[str]:
        out = [f"dataset      {self.name} ({self.path})", f"vocabulary   {self.vocab_size} tokens"]
        for split in SPLITS:
            if self.segments.get(split):
                out.append(
                    f"{split:<12} {self.segments[split]} segments from {self.songs[split]} songs, "
                    f"{self.tokens[split]} tokens"
                )
        if self.length_percentiles:
            out.append(
                "length       " + ", ".join(f"{k}={v}" for k, v in self.length_percentiles.items())
            )
        if self.dropped:
            out.append(
                "dropped      " + ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items()))
            )
        return out


def dataset_dir(artifacts: Path, name: str) -> Path:
    """Folder of the dataset ``name`` under the artifacts directory."""
    return artifacts / "datasets" / name


def build_dataset(
    config: BacklineConfig,
    library: LibraryPaths,
    out_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> DatasetReport:
    """Encode every segment of every dataset song (``ok`` status) into ``out_dir``."""
    vocab = Vocab.from_config(config)
    encoder = SegmentEncoder(config, vocab)
    context = config.model.context_length
    with Catalog(library.catalog) as catalog:
        songs = [s for s in catalog.songs("ok") if s.split in SPLITS]
    buffers: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    records: dict[str, list[tuple[Any, ...]]] = {s: [] for s in SPLITS}
    offsets = dict.fromkeys(SPLITS, 0)
    song_count: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    lengths: list[int] = []
    song_ids: list[str] = []

    for done, song in enumerate(songs, start=1):
        score = load_score(library.score_path(song.id))
        annotation_path = library.annotation_path(song.id)
        if annotation_path.is_file():
            annotation = load_annotation(annotation_path)
        else:
            annotation = annotate_score(score)
            save_annotation(annotation, annotation_path)
        split = str(song.split)
        song_index = len(song_ids)
        song_ids.append(song.id)
        used = False
        for source in song_segments(score, annotation, config):
            try:
                encoded = encoder.encode(source)
            except EncodeError:
                dropped["unencodable"] += 1
                continue
            if len(encoded.tokens) > context:
                dropped["longer_than_context"] += 1
                continue
            tokens = np.asarray(encoded.tokens, dtype=np.uint16)
            buffers[split].append(tokens)
            records[split].append(
                (
                    offsets[split],
                    len(tokens),
                    encoded.target_start,
                    song_index,
                    source.start_bar,
                    source.n_bars,
                    *encoded.guitar_range,
                    *encoded.bass_range,
                )
            )
            offsets[split] += len(tokens)
            lengths.append(len(tokens))
            used = True
        if used:
            song_count[split] += 1
        if progress:
            progress(done, len(songs))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        data = np.concatenate(buffers[split]) if buffers[split] else np.zeros(0, dtype=np.uint16)
        data.tofile(out_dir / f"{split}.tokens")
        np.save(out_dir / f"{split}.index.npy", np.array(records[split], dtype=INDEX_DTYPE))
    (out_dir / "vocab.json").write_text(vocab.to_json(), encoding="utf-8")
    (out_dir / "songs.json").write_text(json.dumps(song_ids), encoding="utf-8")
    percentiles = (
        {f"p{q}": int(np.percentile(lengths, q)) for q in (50, 90, 99)} | {"max": max(lengths)}
        if lengths
        else {}
    )
    report = DatasetReport(
        name=out_dir.name,
        path=str(out_dir),
        songs={s: song_count[s] for s in SPLITS},
        segments={s: len(records[s]) for s in SPLITS},
        tokens={s: offsets[s] for s in SPLITS},
        dropped=dict(dropped),
        length_percentiles=percentiles,
        vocab_size=len(vocab),
    )
    meta = {
        "config": config.name,
        "vocab": vocab.fingerprint(),
        "context_length": context,
        "tokenizer": config.tokenizer.model_dump(mode="json"),
        "report": asdict(report),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return report
