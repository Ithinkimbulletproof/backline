# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Encode a segment of a song as tokens, and decode tokens back to notes and a plan.

Layout::

    BOS META KEY TS_NUM TS_DEN TEMPO
    GUITAR (BAR (POS [VEL] (PITCH DUR [PM] [DEAD])+)*)*  SEP
    PLAN   (BAR [KEY] (POS (DEG QUAL | NC))+)*
    BASS   (BAR (POS [VEL] ([ROLE] PITCH DUR [PM] [DEAD])+)*)*
    DRUMS  (BAR (POS ([VEL] DRUM)+)*)*
    EOS

Every bar of every stream starts with ``BAR``; positions are grid steps from the bar start.
``VEL`` is written only when the level is not ``normal``. Plan degrees are relative to the
key in force (``KEY`` in the header, or a later ``KEY`` after a bar line).
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from backline.config import BacklineConfig
from backline.music import BassRole, ChordQuality, Dynamic, Mode
from backline.score import Note, NoteFlag, Score, TimeSignature
from backline.theory.annotate import Annotation
from backline.theory.key import KeySpan
from backline.tokenizer.vocab import (
    PITCH_RANGE,
    STEP,
    TS_DENOMINATORS,
    TS_NUMERATORS,
    Vocab,
    key_token,
    snap_duration,
    tempo_bucket,
)


@dataclass(frozen=True, slots=True)
class SegmentSource:
    """Which bars of which tracks form a training segment."""

    score: Score
    annotation: Annotation
    guitar: int
    bass: int
    drums: int
    start_bar: int
    n_bars: int


@dataclass(frozen=True, slots=True)
class Encoded:
    tokens: list[int]
    target_start: int
    """Index of the ``PLAN`` token: everything from here on is predicted."""
    guitar_range: tuple[int, int]
    bass_range: tuple[int, int]


class EncodeError(ValueError):
    """Raised when a segment cannot be encoded (meter change, bar too long...)."""


@dataclass(slots=True)
class _Event:
    level: Dynamic
    items: list[list[str]] = field(default_factory=list)


class SegmentEncoder:
    """Turns segments into token ids with a fixed vocabulary."""

    def __init__(self, config: BacklineConfig, vocab: Vocab) -> None:
        self.config = config
        self.vocab = vocab

    # ------------------------------------------------------------ public API

    def encode(self, source: SegmentSource) -> Encoded:
        score = source.score
        bars = range(source.start_bar, source.start_bar + source.n_bars)
        meter, bar_steps = self._meter(score, bars)
        starts = score.bar_starts()
        key = source.annotation.key_at_bar(bars.start)
        tempo = score.tempo_at(starts[bars.start])

        tokens = self.header(key, meter, tempo)
        guitar = score.tracks[source.guitar].notes
        tokens += ["GUITAR", *self._pitched(guitar, starts, bars, bar_steps, roles=None)]
        tokens.append("SEP")
        target = len(tokens)
        tokens += ["PLAN", *self._plan(source.annotation, starts, bars, bar_steps, key)]
        bass_track = score.tracks[source.bass]
        roles = None
        if self.config.tokenizer.bass_role_tokens and source.annotation.bass_roles:
            notes = bass_track.sorted_notes()
            roles = {
                (n.start, n.pitch): r
                for n, r in zip(notes, source.annotation.bass_roles, strict=False)
            }
        tokens += ["BASS", *self._pitched(bass_track.notes, starts, bars, bar_steps, roles=roles)]
        tokens += ["DRUMS", *self._drums(score.tracks[source.drums].notes, starts, bars, bar_steps)]
        tokens.append("EOS")
        return Encoded(
            tokens=[self.vocab.id(t) for t in tokens],
            target_start=target,
            guitar_range=_pitch_range(guitar, starts, bars),
            bass_range=_pitch_range(bass_track.notes, starts, bars),
        )

    def header(self, key: KeySpan, meter: TimeSignature, tempo: float) -> list[str]:
        return [
            "BOS",
            "META",
            key_token(key.tonic, key.mode),
            f"TS_NUM_{meter.numerator}",
            f"TS_DEN_{meter.denominator}",
            f"TEMPO_{tempo_bucket(tempo, self.config.tokenizer.tempo_buckets)}",
        ]

    def condition(self, score: Score, guitar: int, bars: range, key: KeySpan) -> list[int]:
        """``BOS .. SEP`` for a riff: what the model is prompted with at generation time."""
        meter, bar_steps = self._meter(score, bars)
        starts = score.bar_starts()
        tokens = self.header(key, meter, score.tempo_at(starts[bars.start]))
        tokens += [
            "GUITAR",
            *self._pitched(score.tracks[guitar].notes, starts, bars, bar_steps, roles=None),
        ]
        tokens.append("SEP")
        return [self.vocab.id(t) for t in tokens]

    # ------------------------------------------------------------ streams

    def _meter(self, score: Score, bars: range) -> tuple[TimeSignature, int]:
        if bars.stop > score.n_bars:
            raise EncodeError("segment runs past the end of the song")
        meters = {
            (score.time_signature_at(b).numerator, score.time_signature_at(b).denominator)
            for b in bars
        }
        if len(meters) != 1:
            raise EncodeError("meter changes inside the segment")
        meter = score.time_signature_at(bars.start)
        if meter.numerator not in TS_NUMERATORS or meter.denominator not in TS_DENOMINATORS:
            raise EncodeError(f"unsupported meter {meter}")
        ticks = meter.bar_ticks(score.tpq)
        if ticks % STEP or ticks // STEP > self.config.tokenizer.max_positions:
            raise EncodeError(f"bar of {meter} does not fit the position grid")
        return meter, ticks // STEP

    def _locate(
        self, tick: int, starts: list[int], bars: range, bar_steps: int
    ) -> tuple[int, int] | None:
        """(bar offset in segment, step in bar) of a tick, or None outside the segment."""
        step = round((tick - starts[bars.start]) / STEP)
        bar, pos = divmod(step, bar_steps)
        if not 0 <= bar < len(bars):
            return None
        return bar, pos

    def _pitched(
        self,
        notes: Iterable[Note],
        starts: list[int],
        bars: range,
        bar_steps: int,
        roles: dict[tuple[int, int], BassRole] | None,
    ) -> list[str]:
        velocity = self.config.velocity
        segment_end = (len(bars)) * bar_steps
        events: dict[tuple[int, int], _Event] = {}
        for note in sorted(notes):
            where = self._locate(note.start, starts, bars, bar_steps)
            if where is None or not PITCH_RANGE[0] <= note.pitch <= PITCH_RANGE[1]:
                continue
            bar, pos = where
            begin = bar * bar_steps + pos
            length = min(round(note.duration / STEP), segment_end - begin)
            level = velocity.level_of(note.velocity, note.flags)
            event = events.setdefault(where, _Event(level))
            if _LOUDNESS.index(level) > _LOUDNESS.index(event.level):
                event.level = level
            item = []
            if roles is not None:
                role = roles.get((note.start, note.pitch))
                if role is not None:
                    item.append(f"ROLE_{role.value}")
            item += [f"PITCH_{note.pitch}", f"DUR_{snap_duration(length)}"]
            if note.flags & NoteFlag.PALM_MUTE:
                item.append("PM")
            if note.flags & NoteFlag.DEAD:
                item.append("DEAD")
            event.items.append(item)
        return self._bars(events, len(bars))

    def _drums(
        self, notes: Iterable[Note], starts: list[int], bars: range, bar_steps: int
    ) -> list[str]:
        drums_cfg, velocity = self.config.drums, self.config.velocity
        order = {c: i for i, c in enumerate(drums_cfg.classes)}
        hits: dict[tuple[int, int], dict[str, Dynamic]] = defaultdict(dict)
        for note in notes:
            drum_class = drums_cfg.class_of(note.pitch)
            where = self._locate(note.start, starts, bars, bar_steps)
            if drum_class is None or where is None:
                continue
            level = velocity.level_of(note.velocity, note.flags)
            previous = hits[where].get(drum_class)
            if previous is None or _LOUDNESS.index(level) > _LOUDNESS.index(previous):
                hits[where][drum_class] = level
        out: list[str] = []
        by_bar: dict[int, list[tuple[int, dict[str, Dynamic]]]] = defaultdict(list)
        for (bar, pos), classes in hits.items():
            by_bar[bar].append((pos, classes))
        for bar in range(len(bars)):
            out.append("BAR")
            for pos, classes in sorted(by_bar[bar]):
                out.append(f"POS_{pos}")
                for drum_class in sorted(classes, key=order.__getitem__):
                    if classes[drum_class] is not Dynamic.NORMAL:
                        out.append(f"VEL_{classes[drum_class].value}")
                    out.append(f"DRUM_{drum_class}")
        return out

    def _plan(
        self, annotation: Annotation, starts: list[int], bars: range, bar_steps: int, key: KeySpan
    ) -> list[str]:
        chord_starts = [c.start for c in annotation.chords]
        out: list[str] = []
        current = key
        for offset, bar in enumerate(bars):
            out.append("BAR")
            bar_key = annotation.key_at_bar(bar)
            if offset > 0 and (bar_key.tonic, bar_key.mode) != (current.tonic, current.mode):
                out.append(key_token(bar_key.tonic, bar_key.mode))
            current = bar_key
            begin, end = starts[bar], starts[bar + 1]
            index = max(bisect.bisect_right(chord_starts, begin) - 1, 0)
            while index < len(annotation.chords) and annotation.chords[index].start < end:
                chord = annotation.chords[index]
                pos = max(round((max(chord.start, begin) - begin) / STEP), 0)
                if pos < bar_steps:
                    out.append(f"POS_{pos}")
                    if chord.root is None or chord.quality is None:
                        out.append("NC")
                    else:
                        out += [
                            f"DEG_{(chord.root - current.tonic) % 12}",
                            f"QUAL_{chord.quality.value}",
                        ]
                index += 1
        return out

    @staticmethod
    def _bars(events: dict[tuple[int, int], _Event], n_bars: int) -> list[str]:
        out: list[str] = []
        by_bar: dict[int, list[tuple[int, _Event]]] = defaultdict(list)
        for (bar, pos), event in events.items():
            by_bar[bar].append((pos, event))
        for bar in range(n_bars):
            out.append("BAR")
            for pos, event in sorted(by_bar[bar], key=lambda item: item[0]):
                out.append(f"POS_{pos}")
                if event.level is not Dynamic.NORMAL:
                    out.append(f"VEL_{event.level.value}")
                for item in sorted(
                    event.items, key=lambda it: next(t for t in it if t.startswith("PITCH_"))
                ):
                    out += item
        return out


_LOUDNESS = (Dynamic.GHOST, Dynamic.NORMAL, Dynamic.ACCENT, Dynamic.MAX)


def _pitch_range(notes: Iterable[Note], starts: list[int], bars: range) -> tuple[int, int]:
    lo, hi = starts[bars.start], starts[bars.stop]
    pitches = [n.pitch for n in notes if lo <= n.start < hi]
    return (min(pitches), max(pitches)) if pitches else (0, 0)


# ------------------------------------------------------------------- decoding


@dataclass(frozen=True, slots=True)
class DecodedNote:
    bar: int
    pos: int
    pitch: int
    duration: int
    """In grid steps."""
    level: Dynamic
    palm_mute: bool = False
    dead: bool = False
    role: BassRole | None = None


@dataclass(frozen=True, slots=True)
class DecodedHit:
    bar: int
    pos: int
    drum_class: str
    level: Dynamic


@dataclass(frozen=True, slots=True)
class PlanChord:
    bar: int
    pos: int
    degree: int | None
    quality: ChordQuality | None
    tonic: int
    mode: Mode

    @property
    def root(self) -> int | None:
        return None if self.degree is None else (self.tonic + self.degree) % 12


@dataclass(slots=True)
class Decoded:
    """A token sequence read back."""

    tonic: int = 0
    mode: Mode = Mode.MAJOR
    meter: tuple[int, int] = (4, 4)
    tempo_bucket: int = 0
    guitar: list[DecodedNote] = field(default_factory=list)
    plan: list[PlanChord] = field(default_factory=list)
    bass: list[DecodedNote] = field(default_factory=list)
    drums: list[DecodedHit] = field(default_factory=list)


_STREAMS = ("GUITAR", "PLAN", "BASS", "DRUMS")


def decode(ids: Sequence[int], vocab: Vocab) -> Decoded:
    """Read tokens back. Tolerant: tokens that make no sense where they stand are skipped."""
    out = Decoded()
    stream: str | None = None
    bar = pos = -1
    level = Dynamic.NORMAL
    tonic, mode = 0, Mode.MAJOR
    numerator, denominator = 4, 4
    role: BassRole | None = None
    pitch: int | None = None
    degree: int | None = None

    for index in ids:
        token = vocab.token(index)
        placed = bar >= 0 and pos >= 0
        if token in _STREAMS:
            stream, bar, pos, level = token, -1, -1, Dynamic.NORMAL
            if token == "PLAN":
                tonic, mode = out.tonic, out.mode
        elif token == "EOS":
            break
        elif token.startswith("KEY_"):
            _, key_tonic, key_mode = token.split("_")
            if stream is None:
                out.tonic, out.mode = int(key_tonic), Mode(key_mode)
            elif stream == "PLAN":
                tonic, mode = int(key_tonic), Mode(key_mode)
        elif token.startswith("TS_NUM_"):
            numerator = int(token[7:])
        elif token.startswith("TS_DEN_"):
            denominator = int(token[7:])
        elif token.startswith("TEMPO_"):
            out.tempo_bucket = int(token[6:])
        elif token == "BAR":
            bar, pos, level = bar + 1, -1, Dynamic.NORMAL
        elif token.startswith("POS_"):
            pos, level = int(token[4:]), Dynamic.NORMAL
        elif token.startswith("VEL_"):
            level = Dynamic(token[4:])
        elif token.startswith("ROLE_"):
            role = BassRole(token[5:])
        elif token.startswith("PITCH_"):
            pitch = int(token[6:])
        elif (
            token.startswith("DUR_")
            and pitch is not None
            and placed
            and stream in ("GUITAR", "BASS")
        ):
            notes = out.guitar if stream == "GUITAR" else out.bass
            notes.append(DecodedNote(bar, pos, pitch, int(token[4:]), level, role=role))
            pitch = role = None
        elif token in ("PM", "DEAD") and stream in ("GUITAR", "BASS"):
            notes = out.guitar if stream == "GUITAR" else out.bass
            if notes:
                if token == "PM":
                    notes[-1] = replace(notes[-1], palm_mute=True)
                else:
                    notes[-1] = replace(notes[-1], dead=True)
        elif token.startswith("DRUM_") and stream == "DRUMS" and placed:
            out.drums.append(DecodedHit(bar, pos, token[5:], level))
            level = Dynamic.NORMAL
        elif token == "NC" and stream == "PLAN" and placed:
            out.plan.append(PlanChord(bar, pos, None, None, tonic, mode))
        elif token.startswith("DEG_") and stream == "PLAN":
            degree = int(token[4:])
        elif token.startswith("QUAL_") and stream == "PLAN" and degree is not None and placed:
            out.plan.append(PlanChord(bar, pos, degree, ChordQuality(token[5:]), tonic, mode))
            degree = None
    out.meter = (numerator, denominator)
    return out
