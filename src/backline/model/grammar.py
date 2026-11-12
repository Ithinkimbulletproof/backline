# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Sequence grammar for constrained decoding of ``PLAN BASS DRUMS EOS``.

At every step the grammar lists the tokens that keep the sequence valid: each stream has
exactly the riff's number of bars, positions grow within a bar and stay inside it, every
bar of the plan starts with a chord, bass pitches stay inside the target instrument, notes
carry a duration, drum hits name a class. A model sampling only allowed tokens therefore
always produces a part that decodes into notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from backline.tokenizer.vocab import PITCH_RANGE, Vocab


class _Expect(Enum):
    START = auto()
    """Right after SEP: the PLAN marker."""
    STREAM_BAR = auto()
    """Right after a stream marker: a bar line."""
    PLAN_AFTER_BAR = auto()
    PLAN_AFTER_POS = auto()
    PLAN_AFTER_DEG = auto()
    PLAN_AFTER_CHORD = auto()
    BASS_AFTER_BAR = auto()
    BASS_AFTER_POS = auto()
    BASS_AFTER_VEL = auto()
    BASS_AFTER_ROLE = auto()
    BASS_AFTER_PITCH = auto()
    BASS_AFTER_NOTE = auto()
    DRUMS_AFTER_BAR = auto()
    DRUMS_AFTER_POS = auto()
    DRUMS_AFTER_VEL = auto()
    DRUMS_AFTER_HIT = auto()
    DONE = auto()


@dataclass(slots=True)
class GrammarState:
    stream: str = "START"
    expect: _Expect = _Expect.START
    bars: int = 0
    last_pos: int = -1
    flags: set[str] = field(default_factory=set)


class Grammar:
    """Allowed next tokens for the generated part of a sequence."""

    def __init__(
        self, vocab: Vocab, *, n_bars: int, bar_steps: int, bass_range: tuple[int, int]
    ) -> None:
        self.vocab = vocab
        self.n_bars = n_bars
        self.bar_steps = bar_steps
        lo = max(bass_range[0], PITCH_RANGE[0])
        hi = min(bass_range[1], PITCH_RANGE[1])
        ids = vocab.ids
        self.plan = ids["PLAN"]
        self.bar = ids["BAR"]
        self.bass = ids["BASS"]
        self.drums = ids["DRUMS"]
        self.eos = ids["EOS"]
        self.nc = ids["NC"]
        self.pos = [ids[f"POS_{p}"] for p in range(bar_steps)]
        self.pos_value = {token: p for p, token in enumerate(self.pos)}
        self.keys = vocab.ids_with_prefix("KEY_")
        self.degrees = vocab.ids_with_prefix("DEG_")
        self.qualities = vocab.ids_with_prefix("QUAL_")
        self.velocities = vocab.ids_with_prefix("VEL_")
        self.roles = vocab.ids_with_prefix("ROLE_")
        self.pitches = [ids[f"PITCH_{p}"] for p in range(lo, hi + 1)]
        self.durations = vocab.ids_with_prefix("DUR_")
        self.drum_classes = vocab.ids_with_prefix("DRUM_")
        self.pm, self.dead = ids["PM"], ids["DEAD"]

    def start(self) -> GrammarState:
        """State right after ``SEP``."""
        return GrammarState()

    def accepts(self, state: GrammarState, tokens: list[int]) -> GrammarState:
        """Advance through ``tokens`` (e.g. a forced plan), raising if one is not allowed."""
        for token in tokens:
            if token not in self.allowed(state):
                raise ValueError(f"token {self.vocab.token(token)} is not allowed here")
            state = self.advance(state, token)
        return state

    def _next_positions(self, state: GrammarState) -> list[int]:
        return self.pos[state.last_pos + 1 :]

    def _bar_or_end(self, state: GrammarState, end: int) -> list[int]:
        return [self.bar] if state.bars < self.n_bars else [end]

    def allowed(self, state: GrammarState) -> list[int]:
        e = state.expect
        if e is _Expect.START:
            return [self.plan]
        if e is _Expect.DONE:
            return [self.eos]
        if e is _Expect.STREAM_BAR:
            return [self.bar]
        if e is _Expect.PLAN_AFTER_BAR:
            return [self.pos[0], *self.keys]
        if e is _Expect.PLAN_AFTER_POS:
            return [*self.degrees, self.nc]
        if e is _Expect.PLAN_AFTER_DEG:
            return self.qualities
        if e is _Expect.PLAN_AFTER_CHORD:
            return [*self._next_positions(state), *self._bar_or_end(state, self.bass)]
        if e is _Expect.BASS_AFTER_BAR:
            return [*self._next_positions(state), *self._bar_or_end(state, self.drums)]
        if e is _Expect.BASS_AFTER_POS:
            return [*self.velocities, *self.roles, *self.pitches]
        if e is _Expect.BASS_AFTER_VEL:
            return [*self.roles, *self.pitches]
        if e is _Expect.BASS_AFTER_ROLE:
            return self.pitches
        if e is _Expect.BASS_AFTER_PITCH:
            return self.durations
        if e is _Expect.BASS_AFTER_NOTE:
            flags = [
                t for t, name in ((self.pm, "PM"), (self.dead, "DEAD")) if name not in state.flags
            ]
            return [
                *flags,
                *self.roles,
                *self.pitches,
                *self._next_positions(state),
                *self._bar_or_end(state, self.drums),
            ]
        if e is _Expect.DRUMS_AFTER_BAR:
            return [*self._next_positions(state), *self._bar_or_end(state, self.eos)]
        if e is _Expect.DRUMS_AFTER_POS:
            return [*self.velocities, *self.drum_classes]
        if e is _Expect.DRUMS_AFTER_VEL:
            return self.drum_classes
        return [
            *self.velocities,
            *self.drum_classes,
            *self._next_positions(state),
            *self._bar_or_end(state, self.eos),
        ]

    def advance(self, state: GrammarState, token: int) -> GrammarState:
        """State after ``token`` (assumed allowed)."""
        s = GrammarState(state.stream, state.expect, state.bars, state.last_pos, set(state.flags))
        if token == self.plan:
            return GrammarState("PLAN", _Expect.STREAM_BAR)
        if token == self.bar:
            s.bars += 1
            s.last_pos = -1
            s.expect = {
                "PLAN": _Expect.PLAN_AFTER_BAR,
                "BASS": _Expect.BASS_AFTER_BAR,
                "DRUMS": _Expect.DRUMS_AFTER_BAR,
            }[s.stream]
            return s
        if token == self.bass:
            return GrammarState("BASS", _Expect.STREAM_BAR)
        if token == self.drums:
            return GrammarState("DRUMS", _Expect.STREAM_BAR)
        if token == self.eos:
            return GrammarState("END", _Expect.DONE)
        if token in self.pos_value:
            s.last_pos = self.pos_value[token]
            s.flags = set()
            s.expect = {
                "PLAN": _Expect.PLAN_AFTER_POS,
                "BASS": _Expect.BASS_AFTER_POS,
                "DRUMS": _Expect.DRUMS_AFTER_POS,
            }[s.stream]
            return s
        name = self.vocab.token(token)
        if s.stream == "PLAN":
            if name.startswith("KEY_"):
                s.expect = _Expect.PLAN_AFTER_BAR
            elif name.startswith("DEG_"):
                s.expect = _Expect.PLAN_AFTER_DEG
            elif name.startswith("QUAL_") or token == self.nc:
                s.expect = _Expect.PLAN_AFTER_CHORD
        elif s.stream == "BASS":
            if name.startswith("VEL_"):
                s.expect = _Expect.BASS_AFTER_VEL
            elif name.startswith("ROLE_"):
                s.expect = _Expect.BASS_AFTER_ROLE
                s.flags = set()
            elif name.startswith("PITCH_"):
                s.expect = _Expect.BASS_AFTER_PITCH
                s.flags = set()
            elif name.startswith("DUR_"):
                s.expect = _Expect.BASS_AFTER_NOTE
            elif name in ("PM", "DEAD"):
                s.flags.add(name)
                s.expect = _Expect.BASS_AFTER_NOTE
        elif s.stream == "DRUMS":
            if name.startswith("VEL_"):
                s.expect = _Expect.DRUMS_AFTER_VEL
            elif name.startswith("DRUM_"):
                s.expect = _Expect.DRUMS_AFTER_HIT
        return s
