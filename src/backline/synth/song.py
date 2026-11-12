# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble synthetic songs: sections, chord spans, the three parts and the ground truth."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

from backline.config import BacklineConfig, StringInstrument
from backline.music import BassRole, ChordQuality, Dynamic, Mode
from backline.score import TPQ, Note, NoteFlag, Score, Tempo, TimeSignature, Track, TrackRole
from backline.synth import names
from backline.synth.harmony import Progression, progressions_for
from backline.synth.parts import (
    BASS_STYLES_FULL,
    BASS_STYLES_POWER,
    BASS_STYLES_SOUNDING_THIRD,
    FULL_CHORD_RIFFS,
    GROOVES,
    RIFF_STYLES,
    Groove,
    Span,
    bass_hits,
    drum_bar,
    guitar_pitches,
    riff_hits,
)
from backline.timing import swing_tick

_FLAGS_FOR_DYNAMIC = {
    Dynamic.GHOST: NoteFlag.GHOST,
    Dynamic.NORMAL: NoteFlag.NONE,
    Dynamic.ACCENT: NoteFlag.ACCENT,
    Dynamic.MAX: NoteFlag.HEAVY_ACCENT,
}
_SECTION_ORDER = ("A", "B", "A", "B", "C", "B")
_SECTION_NAMES = {"A": "verse", "B": "chorus", "C": "bridge"}


@dataclass(frozen=True, slots=True)
class SectionTruth:
    """How a section was built."""

    name: str
    start_bar: int
    n_bars: int
    tonic: int
    mode: Mode
    progression: str
    riff: str
    bass: str
    groove: str
    full_chords: bool


@dataclass(frozen=True, slots=True)
class ChordTruth:
    """The chord that sounds over ``start..end`` ticks.

    ``quality`` is what is audible: ``POWER`` when neither the guitar voicing nor the
    bass line plays the third, otherwise the template quality.
    """

    start: int
    end: int
    root: int
    degree: int
    quality: ChordQuality


@dataclass(frozen=True, slots=True)
class BassNoteTruth:
    """Role of the bass note that starts at ``start`` with ``pitch``."""

    start: int
    pitch: int
    role: BassRole


@dataclass(slots=True)
class SynthSong:
    """A generated song and the labels it was generated from."""

    score: Score
    meter: str
    swing: bool
    sections: list[SectionTruth] = field(default_factory=list)
    chords: list[ChordTruth] = field(default_factory=list)
    bass_notes: list[BassNoteTruth] = field(default_factory=list)

    def truth_dict(self) -> dict[str, Any]:
        """Ground truth as plain JSON-serialisable data."""
        return {
            "title": self.score.title,
            "artist": self.score.artist,
            "tpq": self.score.tpq,
            "meter": self.meter,
            "swing": self.swing,
            "tempo": self.score.tempos[0].bpm,
            "sections": [asdict(s) for s in self.sections],
            "chords": [asdict(c) for c in self.chords],
            "bass_notes": [asdict(b) for b in self.bass_notes],
        }


@dataclass(frozen=True, slots=True)
class _Design:
    progression: Progression
    n_bars: int
    riff: str
    bass: str
    groove: Groove
    full: bool
    backbeat: Dynamic
    open_hat: bool
    blue: bool


@dataclass(frozen=True, slots=True)
class _PlannedSpan:
    start: int
    end: int
    root_pc: int
    degree: int
    quality: ChordQuality
    tonic: int
    mode: Mode
    section: int


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _make_design(
    rng: random.Random,
    *,
    family: str,
    meter: str,
    swing: bool,
    config: BacklineConfig,
) -> _Design:
    choices = progressions_for(family, whole_bars_only=meter != "4/4")
    if meter != "4/4":
        choices = [p for p in choices if not p.blues]
    elif swing and family == "major" and rng.random() < 0.5:
        choices = [p for p in choices if p.blues]
    progression = rng.choice(choices)
    cycle = progression.cycle_bars
    n_bars = cycle if cycle >= 8 else rng.choice([b for b in (4, 8) if b % cycle == 0])

    full = rng.random() < config.synth.full_chord_probability
    if progression.blues:
        full = False
        riff = rng.choice(("boogie", "eighths_pm", "quarters"))
        bass = rng.choice(("blues_boogie", "blues_boogie", "root_eighths", "follow_riff"))
    else:
        riff = rng.choice(FULL_CHORD_RIFFS if full else RIFF_STYLES)
        bass = rng.choice(BASS_STYLES_FULL if full else BASS_STYLES_POWER)
    if bass == "follow_riff" and riff == "sustain":
        bass = "root_eighths"

    groove_family = "shuffle" if swing else meter
    return _Design(
        progression=progression,
        n_bars=n_bars,
        riff=riff,
        bass=bass,
        groove=rng.choice(GROOVES[groove_family]),
        full=full,
        backbeat=rng.choice((Dynamic.NORMAL, Dynamic.ACCENT, Dynamic.MAX)),
        open_hat=rng.random() < 0.3,
        blue=rng.random() < 0.5,
    )


def generate_song(
    config: BacklineConfig,
    rng: random.Random,
    *,
    title: str,
    artist: str,
) -> SynthSong:
    """Generate one song with guitar, bass and drums plus its ground truth."""
    synth = config.synth
    meter = _weighted(rng, synth.time_signatures)
    numerator, denominator = (int(x) for x in meter.split("/"))
    time_signature = TimeSignature(0, numerator, denominator)
    bar_ticks = time_signature.bar_ticks(TPQ)
    swing = meter == "4/4" and rng.random() < synth.shuffle_probability
    tempo = rng.randint(*synth.tempo_range)
    family = rng.choice(("major", "minor"))
    tonic = rng.randrange(12)
    profile: StringInstrument = rng.choice(list(config.instruments.bass.profiles.values()))

    n_sections = rng.randint(*synth.sections_per_song)
    order = [_SECTION_ORDER[i % len(_SECTION_ORDER)] for i in range(n_sections)]
    designs = {
        letter: _make_design(rng, family=family, meter=meter, swing=swing, config=config)
        for letter in sorted(set(order))
    }
    key_change = n_sections >= 3 and rng.random() < synth.key_change_probability

    # Pass 1: lay out sections and chord spans.
    sections: list[SectionTruth] = []
    planned: list[_PlannedSpan] = []
    bar = 0
    for index, letter in enumerate(order):
        design = designs[letter]
        section_tonic = (tonic + 2) % 12 if key_change and index == len(order) - 1 else tonic
        sections.append(
            SectionTruth(
                name=_SECTION_NAMES[letter],
                start_bar=bar,
                n_bars=design.n_bars,
                tonic=section_tonic,
                mode=design.progression.mode,
                progression=design.progression.name,
                riff=design.riff,
                bass=design.bass,
                groove=design.groove.name,
                full_chords=design.full,
            )
        )
        tick = bar * bar_ticks
        section_end = (bar + design.n_bars) * bar_ticks
        while tick < section_end:
            for step in design.progression.steps:
                length = int(step.bars * Fraction(bar_ticks))
                planned.append(
                    _PlannedSpan(
                        start=tick,
                        end=tick + length,
                        root_pc=(section_tonic + step.degree) % 12,
                        degree=step.degree,
                        quality=step.quality,
                        tonic=section_tonic,
                        mode=design.progression.mode,
                        section=index,
                    )
                )
                tick += length
        bar += design.n_bars
    n_bars = bar

    # Pass 2: write the parts span by span, knowing which chord comes next.
    guitar: list[Note] = []
    bass: list[Note] = []
    roles: list[BassNoteTruth] = []
    chords: list[ChordTruth] = []
    for i, plan in enumerate(planned):
        design = designs[order[plan.section]]
        following = planned[i + 1] if i + 1 < len(planned) else planned[0]
        span = Span(
            start=plan.start,
            end=plan.end,
            root_pc=plan.root_pc,
            degree=plan.degree,
            quality=plan.quality,
            tonic_pc=plan.tonic,
            mode=plan.mode,
            next_root_pc=following.root_pc,
        )
        hits = riff_hits(design.riff, span, full=design.full)
        for hit in hits:
            for pitch in guitar_pitches(hit.shape, span.root_pc, span.quality):
                guitar.append(
                    Note(
                        start=span.start + hit.offset,
                        pitch=pitch,
                        duration=hit.duration,
                        velocity=config.velocity.velocity_of(hit.dynamic),
                        flags=hit.flags | _FLAGS_FOR_DYNAMIC[hit.dynamic],
                    )
                )
        for bhit in bass_hits(
            design.bass,
            span,
            low=profile.lowest,
            high=profile.highest,
            riff=hits,
            blue=design.blue,
        ):
            start = span.start + bhit.offset
            bass.append(
                Note(
                    start=start,
                    pitch=bhit.pitch,
                    duration=bhit.duration,
                    velocity=config.velocity.velocity_of(bhit.dynamic),
                    flags=_FLAGS_FOR_DYNAMIC[bhit.dynamic],
                )
            )
            roles.append(BassNoteTruth(start, bhit.pitch, bhit.role))
        third_audible = design.full or design.bass in BASS_STYLES_SOUNDING_THIRD
        chords.append(
            ChordTruth(
                start=plan.start,
                end=plan.end,
                root=plan.root_pc,
                degree=plan.degree,
                quality=plan.quality if third_audible else ChordQuality.POWER,
            )
        )

    drums = _drum_track(config, rng, sections, designs, order, bar_ticks)

    if swing:
        guitar = [_swung(n) for n in guitar]
        bass = [_swung(n) for n in bass]
        roles = [BassNoteTruth(swing_tick(r.start), r.pitch, r.role) for r in roles]

    score = Score(
        n_bars=n_bars,
        time_signatures=[time_signature],
        tempos=[Tempo(0, float(tempo))],
        tracks=[
            Track(
                name=rng.choice(("Guitar", "Rhythm Guitar", "Distortion Guitar", "Gtr. 1")),
                role=TrackRole.GUITAR,
                notes=sorted(guitar),
                program=rng.choice((29, 30)),
                tuning=config.instruments.guitar.tuning,
            ),
            Track(
                name=rng.choice(("Bass", "Electric Bass", "Bass Guitar")),
                role=TrackRole.BASS,
                notes=sorted(bass),
                program=rng.choice((33, 34)),
                tuning=profile.tuning,
            ),
            Track(
                name=rng.choice(("Drums", "Drum Kit", "Kit")),
                role=TrackRole.DRUMS,
                notes=sorted(drums),
                is_percussion=True,
            ),
        ],
        title=title,
        artist=artist,
    )
    return SynthSong(
        score=score,
        meter=meter,
        swing=swing,
        sections=sections,
        chords=chords,
        bass_notes=sorted(roles, key=lambda r: (r.start, r.pitch)),
    )


def _drum_track(
    config: BacklineConfig,
    rng: random.Random,
    sections: list[SectionTruth],
    designs: dict[str, _Design],
    order: list[str],
    bar_ticks: int,
) -> list[Note]:
    drums_cfg = config.drums
    notes: list[Note] = []
    crashes = ("crash1", "crash2")
    for index, section in enumerate(sections):
        design = designs[order[index]]
        is_last = index == len(sections) - 1
        use_china = section.name == "chorus" and rng.random() < 0.3
        for offset in range(section.n_bars):
            bar = section.start_bar + offset
            cymbal: str | None = None
            if offset == 0:
                cymbal = "china" if use_china else crashes[index % 2]
            elif offset == section.n_bars // 2 and rng.random() < 0.2:
                cymbal = "splash"
            fill = offset == section.n_bars - 1 and not is_last and rng.random() < 0.6
            for hit in drum_bar(
                design.groove,
                cymbal=cymbal,
                fill=fill,
                open_hat=design.open_hat and offset % 2 == 1,
                backbeat=design.backbeat,
            ):
                notes.append(
                    Note(
                        start=bar * bar_ticks + hit.offset,
                        pitch=drums_cfg.canonical_note(hit.drum_class),
                        duration=hit.duration,
                        velocity=config.velocity.velocity_of(hit.dynamic),
                        flags=_FLAGS_FOR_DYNAMIC[hit.dynamic],
                    )
                )
    return notes


def _swung(note: Note) -> Note:
    start = swing_tick(note.start)
    end = swing_tick(note.end)
    return Note(
        start=start,
        pitch=note.pitch,
        duration=max(end - start, TPQ // 24),
        velocity=note.velocity,
        flags=note.flags,
    )


def generate_corpus(
    config: BacklineConfig,
    *,
    n_songs: int | None = None,
    seed: int | None = None,
) -> list[SynthSong]:
    """Generate a reproducible corpus; song ``i`` depends only on the seed and ``i``."""
    count = config.synth.n_songs if n_songs is None else n_songs
    base_seed = config.seed if seed is None else seed
    artist_rng = random.Random(f"{base_seed}:artists")
    artists = [names.artist_name(artist_rng) for _ in range(config.synth.n_artists)]
    songs = []
    for index in range(count):
        rng = random.Random(f"{base_seed}:song:{index}")
        artist = artists[rng.randrange(len(artists))]
        songs.append(generate_song(config, rng, title=names.song_title(rng), artist=artist))
    return songs
