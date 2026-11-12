# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Annotate every imported song and summarise the corpus' harmony."""

from __future__ import annotations

import bisect
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from backline.formats.store import load_score
from backline.ingest.catalog import Catalog
from backline.ingest.pipeline import LibraryPaths
from backline.score import Score
from backline.theory.annotate import Annotation, annotate_score, load_annotation, save_annotation
from backline.theory.grid import beat_grid

ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class CorpusHarmony:
    """Harmony statistics of the annotated corpus, kept as reference values for metrics."""

    songs: int = 0
    modes: dict[str, int] = field(default_factory=dict)
    """Key spans per mode."""
    mode_certain_share: float = 0.0
    """Share of key spans in which thirds are actually heard."""
    key_changes_per_song: float = 0.0
    power_chord_share: float = 0.0
    """Share of beats whose chord has no audible third."""
    functions: dict[str, float] = field(default_factory=dict)
    """Share of beats per harmonic function."""
    bass_roles: dict[str, float] = field(default_factory=dict)
    """Share of bass notes per role, over the train split."""


@dataclass(slots=True)
class AnnotateReport:
    annotated: int
    reused: int
    harmony: CorpusHarmony
    report_path: str = ""

    def lines(self) -> list[str]:
        h = self.harmony
        out = [
            f"annotated    {self.annotated} (unchanged {self.reused})",
            f"songs        {h.songs}",
            "modes        " + ", ".join(f"{k}={v}" for k, v in sorted(h.modes.items())),
            f"thirds heard {h.mode_certain_share:.0%} of key spans",
            f"power chords {h.power_chord_share:.0%} of beats",
            "functions    " + ", ".join(f"{k}={v:.0%}" for k, v in sorted(h.functions.items())),
            "bass roles   " + ", ".join(f"{k}={v:.0%}" for k, v in sorted(h.bass_roles.items())),
        ]
        if self.report_path:
            out.append(f"report       {self.report_path}")
        return out


class _Tally:
    def __init__(self) -> None:
        self.songs = 0
        self.modes: Counter[str] = Counter()
        self.spans = self.certain = self.changes = 0
        self.beats = self.power = 0
        self.functions: Counter[str] = Counter()
        self.roles: Counter[str] = Counter()

    def add(self, score: Score, annotation: Annotation, *, train: bool) -> None:
        self.songs += 1
        self.spans += len(annotation.keys)
        self.changes += len(annotation.keys) - 1
        for key in annotation.keys:
            self.modes[key.mode.value] += 1
            self.certain += key.mode_certain
        labelled = annotation.labelled_chords()
        starts = [c.start for c in labelled]
        for beat in beat_grid(score):
            chord = labelled[bisect.bisect_right(starts, beat.start) - 1]
            self.beats += 1
            self.functions[chord.function.value if chord.function else "none"] += 1
            self.power += chord.quality is not None and not chord.quality.has_third
        if train:
            self.roles.update(r.value for r in annotation.bass_roles)

    def summary(self) -> CorpusHarmony:
        def shares(counter: Counter[str]) -> dict[str, float]:
            total = sum(counter.values())
            return {k: round(v / total, 4) for k, v in counter.items()} if total else {}

        return CorpusHarmony(
            songs=self.songs,
            modes=dict(self.modes),
            mode_certain_share=round(self.certain / self.spans, 4) if self.spans else 0.0,
            key_changes_per_song=round(self.changes / self.songs, 3) if self.songs else 0.0,
            power_chord_share=round(self.power / self.beats, 4) if self.beats else 0.0,
            functions=shares(self.functions),
            bass_roles=shares(self.roles),
        )


def annotate_library(
    library: LibraryPaths, *, force: bool = False, progress: ProgressCallback | None = None
) -> AnnotateReport:
    """Annotate imported songs whose annotation is missing or older than the stored song.

    Duplicates are annotated too (the originality checks calibrate on them) but only
    dataset songs count towards the corpus statistics.
    """
    with Catalog(library.catalog) as catalog:
        songs = [s for s in catalog.songs() if s.status in ("ok", "duplicate")]
    annotated = reused = 0
    tally = _Tally()
    for done, song in enumerate(songs, start=1):
        score_path = library.score_path(song.id)
        target = library.annotation_path(song.id)
        score = load_score(score_path)
        fresh = target.is_file() and target.stat().st_mtime >= score_path.stat().st_mtime
        if fresh and not force:
            annotation = load_annotation(target)
            reused += 1
        else:
            annotation = annotate_score(score)
            save_annotation(annotation, target)
            annotated += 1
        if song.status == "ok":
            tally.add(score, annotation, train=song.split == "train")
        if progress:
            progress(done, len(songs))

    report = AnnotateReport(annotated=annotated, reused=reused, harmony=tally.summary())
    library.reports.mkdir(parents=True, exist_ok=True)
    path = library.reports / "annotate_latest.json"
    report.report_path = str(path)
    path.write_text(json.dumps(asdict(report), indent=1), encoding="utf-8")
    return report
