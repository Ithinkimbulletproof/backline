# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Role of every bass note against the chord it sounds over.

Rules, in order:

1. a chord tone is a chord tone (for a power chord: root and fifth);
2. a minor third over a chord with a major third, a flat fifth over any chord, or the
   blues b3/b5 of a major-family key over a power chord is a blue note;
3. a step away that returns to the same note is a neighbour note;
4. a semitone into a chord tone of the next note's chord is a chromatic approach;
5. a step between two steps in the same direction is a passing note;
6. anything else counts as passing when it belongs to the key, as an approach otherwise.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence

from backline.music import BassRole, ChordQuality, Mode
from backline.score import Note
from backline.theory.chords import ChordSpan, chord_tones
from backline.theory.key import KeySpan

_MAJOR_THIRD_QUALITIES = frozenset(
    {ChordQuality.MAJOR, ChordQuality.DOMINANT7, ChordQuality.MAJOR7, ChordQuality.AUGMENTED}
)


def bass_roles(
    notes: Sequence[Note],
    chords: Sequence[ChordSpan],
    key_at: Callable[[int], KeySpan],
) -> list[BassRole]:
    """Roles of ``notes`` (sorted by start); ``key_at`` maps a tick to the key there."""
    starts = [c.start for c in chords]

    def chord_at(tick: int) -> ChordSpan | None:
        index = bisect.bisect_right(starts, tick) - 1
        if index < 0 or chords[index].end <= tick:
            return None
        return chords[index]

    def tones_at(tick: int) -> frozenset[int]:
        chord = chord_at(tick)
        if chord is None or chord.root is None or chord.quality is None:
            return frozenset()
        return chord_tones(chord.root, chord.quality)

    roles = []
    for i, note in enumerate(notes):
        pc = note.pitch % 12
        chord = chord_at(note.start)
        prev = notes[i - 1] if i > 0 else None
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        key = key_at(note.start)
        degree = (pc - key.tonic) % 12

        if chord is None or chord.root is None or chord.quality is None:
            roles.append(BassRole.CHORD_TONE if degree in key.mode.scale else BassRole.PASSING)
            continue
        if pc in chord_tones(chord.root, chord.quality):
            roles.append(BassRole.CHORD_TONE)
            continue
        interval = (pc - chord.root) % 12
        major_family = key.mode in (Mode.MAJOR, Mode.MIXOLYDIAN)
        if (
            (interval == 3 and chord.quality in _MAJOR_THIRD_QUALITIES)
            or interval == 6
            or (major_family and chord.quality is ChordQuality.POWER and degree in (3, 6))
        ):
            roles.append(BassRole.BLUE_NOTE)
            continue
        step_in = abs(note.pitch - prev.pitch) if prev else 0
        step_out = abs(nxt.pitch - note.pitch) if nxt else 0
        if prev and nxt and prev.pitch == nxt.pitch and 1 <= step_in <= 2:
            roles.append(BassRole.NEIGHBOR)
            continue
        if nxt and step_out == 1 and nxt.pitch % 12 in tones_at(nxt.start):
            roles.append(BassRole.CHROMATIC_APPROACH)
            continue
        if (
            prev
            and nxt
            and 1 <= step_in <= 2
            and 1 <= step_out <= 2
            and (note.pitch - prev.pitch) * (nxt.pitch - note.pitch) > 0
        ):
            roles.append(BassRole.PASSING)
            continue
        roles.append(BassRole.PASSING if degree in key.mode.scale else BassRole.CHROMATIC_APPROACH)
    return roles
