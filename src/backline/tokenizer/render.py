# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Turn decoded tokens into a :class:`~backline.score.Score` and plan tokens back from chords."""

from __future__ import annotations

from collections.abc import Sequence

from backline.config import BacklineConfig
from backline.drums import DrumMap
from backline.music import Dynamic, Mode
from backline.score import Note, NoteFlag, Score, Tempo, TimeSignature, Track, TrackRole
from backline.tokenizer.encode import Decoded, DecodedNote, PlanChord
from backline.tokenizer.vocab import STEP, Vocab, key_token

_FLAGS = {
    Dynamic.GHOST: NoteFlag.GHOST,
    Dynamic.NORMAL: NoteFlag.NONE,
    Dynamic.ACCENT: NoteFlag.ACCENT,
    Dynamic.MAX: NoteFlag.HEAVY_ACCENT,
}


def bucket_tempo(bucket: int, edges: tuple[int, ...]) -> float:
    """A representative tempo for a bucket (its middle, or its edge at the extremes)."""
    if bucket <= 0:
        return float(edges[0])
    if bucket >= len(edges):
        return float(edges[-1] + 20)
    return (edges[bucket - 1] + edges[bucket]) / 2


def to_score(
    decoded: Decoded,
    config: BacklineConfig,
    n_bars: int,
    *,
    tempo: float | None = None,
    drum_map: DrumMap | None = None,
    include_guitar: bool = False,
) -> Score:
    """Bass and drum tracks (and optionally the riff) of a decoded segment.

    Drum classes become the drum map's notes when one is given, General MIDI otherwise.
    """
    meter = TimeSignature(0, *decoded.meter)
    bar_steps = meter.bar_ticks() // STEP
    velocity = config.velocity

    def pitched(notes: Sequence[DecodedNote]) -> list[Note]:
        out = []
        for n in notes:
            flags = _FLAGS[n.level]
            if n.palm_mute:
                flags |= NoteFlag.PALM_MUTE
            if n.dead:
                flags |= NoteFlag.DEAD
            start = (n.bar * bar_steps + n.pos) * STEP
            out.append(
                Note(start, n.pitch, n.duration * STEP, velocity.velocity_of(n.level), flags)
            )
        return sorted(out)

    drums = []
    for hit in decoded.drums:
        pitch = (
            drum_map.note_for(hit.drum_class, hit.level)
            if drum_map is not None
            else config.drums.canonical_note(hit.drum_class)
        )
        drums.append(
            Note(
                (hit.bar * bar_steps + hit.pos) * STEP,
                pitch,
                STEP * 3,
                velocity.velocity_of(hit.level),
                _FLAGS[hit.level],
            )
        )
    profile = config.instruments.bass.active
    tracks = [
        Track("Bass", TrackRole.BASS, pitched(decoded.bass), 33, tuning=profile.tuning),
        Track("Drums", TrackRole.DRUMS, sorted(drums), 0, is_percussion=True),
    ]
    if include_guitar:
        tracks.insert(
            0,
            Track(
                "Guitar",
                TrackRole.GUITAR,
                pitched(decoded.guitar),
                30,
                tuning=config.instruments.guitar.tuning,
            ),
        )
    bpm = (
        tempo
        if tempo is not None
        else bucket_tempo(decoded.tempo_bucket, config.tokenizer.tempo_buckets)
    )
    return Score(n_bars=n_bars, time_signatures=[meter], tempos=[Tempo(0, bpm)], tracks=tracks)


def plan_tokens(
    plan: Sequence[PlanChord], n_bars: int, key: tuple[int, Mode], vocab: Vocab
) -> list[int]:
    """``PLAN BAR ...`` tokens for an (edited) plan, ready to force before the parts.

    Every bar must have a chord at position 0; key changes are written after the bar line.
    """
    tokens = ["PLAN"]
    current = key
    by_bar: dict[int, list[PlanChord]] = {}
    for chord in plan:
        by_bar.setdefault(chord.bar, []).append(chord)
    for bar in range(n_bars):
        tokens.append("BAR")
        chords = sorted(by_bar.get(bar, []), key=lambda c: c.pos)
        if not chords or chords[0].pos != 0:
            raise ValueError(f"bar {bar + 1} of the plan has no chord on its first beat")
        if (chords[0].tonic, chords[0].mode) != current:
            current = (chords[0].tonic, chords[0].mode)
            tokens.append(key_token(*current))
        last = -1
        for chord in chords:
            if chord.pos <= last:
                continue
            last = chord.pos
            tokens.append(f"POS_{chord.pos}")
            if chord.degree is None or chord.quality is None:
                tokens.append("NC")
            else:
                tokens += [f"DEG_{chord.degree % 12}", f"QUAL_{chord.quality.value}"]
    tokens.append("BASS")
    return [vocab.id(t) for t in tokens]
