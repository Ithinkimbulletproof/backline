# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Minimal Guitar Pro 7 (``.gp``) writer for synthetic fixtures.

It writes the subset of the GPIF format that :mod:`backline.formats.gpif` reads, laid out
the way Guitar Pro lays it out (id-linked tables, flattened drum articulations, ties,
accent bits, palm mutes). It exists so the ``.gp`` import path can be tested end to end
without real tabs; it does not try to produce files that Guitar Pro itself will open.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Final
from xml.etree.ElementTree import Element, SubElement, tostring

from backline.formats.gpif import GPIF_ENTRY
from backline.formats.tablature import (
    DEFAULT_TUNINGS,
    NoteValue,
    Placed,
    TabBeat,
    TabNote,
    layout_bar,
    nearest_dynamic,
    place_fretted,
    place_percussion,
)
from backline.score import NoteFlag, Score, Track, TrackRole

_INSTRUMENT_TYPES: Final = {
    TrackRole.GUITAR: "electricGuitar",
    TrackRole.BASS: "electricBass",
    TrackRole.DRUMS: "drumKit",
    TrackRole.OTHER: "electricGuitar",
}
_NOTE_VALUE_NAMES: Final = {
    1: "Whole",
    2: "Half",
    4: "Quarter",
    8: "Eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
}
_GM_DRUMS: Final = {
    35: ("Kick", "Acoustic"),
    36: ("Kick", "Hit"),
    37: ("Snare", "Side Stick"),
    38: ("Snare", "Hit"),
    40: ("Snare", "Electric"),
    41: ("Floor Tom", "Low"),
    42: ("Hi-Hat", "Closed"),
    43: ("Floor Tom", "High"),
    44: ("Hi-Hat", "Pedal"),
    45: ("Tom", "Low"),
    46: ("Hi-Hat", "Open"),
    47: ("Tom", "Low-Mid"),
    48: ("Tom", "Hi-Mid"),
    49: ("Crash", "Crash 1"),
    50: ("Tom", "High"),
    51: ("Ride", "Bow"),
    52: ("China", "Hit"),
    53: ("Ride", "Bell"),
    55: ("Splash", "Hit"),
    57: ("Crash", "Crash 2"),
    59: ("Ride", "Ride 2"),
}
_ACCENT_BITS: Final = ((NoteFlag.HEAVY_ACCENT, 0x04), (NoteFlag.ACCENT, 0x08))
_LOUDNESS_FLAGS: Final = NoteFlag.GHOST | NoteFlag.ACCENT | NoteFlag.HEAVY_ACCENT


def write_gp(score: Score, path: Path) -> None:
    """Write ``score`` as a minimal ``.gp`` archive."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("VERSION", "7.0")
        archive.writestr(GPIF_ENTRY, score_to_gpif(score))


def score_to_gpif(score: Score) -> bytes:
    """Serialise ``score`` as a GPIF XML document."""
    return _Writer(score).build()


def _sub(parent: Element, tag: str, text: str | None = None, **attributes: str) -> Element:
    node = SubElement(parent, tag, attributes)
    if text is not None:
        node.text = text
    return node


class _Writer:
    def __init__(self, score: Score) -> None:
        self.score = score
        self.starts = score.bar_starts()
        self.root = Element("GPIF")
        self.bars_el: Element
        self.voices_el: Element
        self.beats_el: Element
        self.notes_el: Element
        self.rhythms_el: Element
        self.rhythm_ids: dict[tuple[int, bool, bool], int] = {}
        self.counts = {"bar": 0, "voice": 0, "beat": 0, "note": 0}

    def _next(self, kind: str) -> int:
        value = self.counts[kind]
        self.counts[kind] += 1
        return value

    def build(self) -> bytes:
        root, score = self.root, self.score
        _sub(root, "GPVersion", "7")
        head = _sub(root, "Score")
        _sub(head, "Title", score.title)
        _sub(head, "Artist", score.artist)

        master = _sub(root, "MasterTrack")
        _sub(master, "Tracks", " ".join(str(i) for i in range(len(score.tracks))))
        automations = _sub(master, "Automations")
        for tempo in score.tempos:
            if tempo.tick >= self.starts[-1]:
                continue
            bar = score.bar_of(tempo.tick)
            length = self.starts[bar + 1] - self.starts[bar]
            automation = _sub(automations, "Automation")
            _sub(automation, "Type", "Tempo")
            _sub(automation, "Linear", "false")
            _sub(automation, "Bar", str(bar))
            _sub(automation, "Position", f"{(tempo.tick - self.starts[bar]) / length:.6g}")
            _sub(automation, "Visible", "true")
            _sub(automation, "Value", f"{tempo.bpm:g} 2")

        tracks_el = _sub(root, "Tracks")
        masters_el = _sub(root, "MasterBars")
        self.bars_el = _sub(root, "Bars")
        self.voices_el = _sub(root, "Voices")
        self.beats_el = _sub(root, "Beats")
        self.notes_el = _sub(root, "Notes")
        self.rhythms_el = _sub(root, "Rhythms")

        bar_ids: list[list[int]] = []
        for index, track in enumerate(score.tracks):
            articulations = self._track(tracks_el, index, track)
            bar_ids.append(self._track_bars(track, articulations))

        for bar in range(score.n_bars):
            ts = score.time_signature_at(bar)
            master_bar = _sub(masters_el, "MasterBar")
            key = _sub(master_bar, "Key")
            _sub(key, "AccidentalCount", "0")
            _sub(key, "Mode", "Major")
            _sub(master_bar, "Time", f"{ts.numerator}/{ts.denominator}")
            _sub(master_bar, "Bars", " ".join(str(ids[bar]) for ids in bar_ids))
        document: bytes = tostring(root, encoding="utf-8", xml_declaration=True)
        return document

    def _track(self, parent: Element, index: int, track: Track) -> dict[int, int]:
        element = _sub(parent, "Track", id=str(index))
        _sub(element, "Name", track.name)
        instrument = _sub(element, "InstrumentSet")
        _sub(instrument, "Name", track.name)
        _sub(
            instrument,
            "Type",
            _INSTRUMENT_TYPES[TrackRole.DRUMS if track.is_percussion else track.role],
        )
        _sub(instrument, "LineCount", "5")
        articulations: dict[int, int] = {}
        if track.is_percussion:
            elements = _sub(instrument, "Elements")
            for pitch in sorted({n.pitch for n in track.notes}):
                name, label = _GM_DRUMS.get(pitch, ("Percussion", str(pitch)))
                drum = _sub(elements, "Element")
                _sub(drum, "Name", name)
                articulation = _sub(_sub(drum, "Articulations"), "Articulation")
                _sub(articulation, "Name", label)
                _sub(articulation, "InputMidiNumbers", str(pitch))
                _sub(articulation, "OutputMidiNumber", str(pitch))
                articulations[pitch] = len(articulations)
        midi = _sub(_sub(_sub(element, "Sounds"), "Sound"), "MIDI")
        _sub(midi, "Program", str(track.program))
        connection = _sub(element, "MidiConnection")
        _sub(connection, "PrimaryChannel", "9" if track.is_percussion else str(index * 2 % 9))
        staff = _sub(_sub(element, "Staves"), "Staff")
        properties = _sub(staff, "Properties")
        if not track.is_percussion:
            tuning = track.tuning or DEFAULT_TUNINGS[track.role]
            prop = _sub(properties, "Property", name="Tuning")
            _sub(prop, "Pitches", " ".join(str(p) for p in tuning))
            _sub(_sub(properties, "Property", name="CapoFret"), "Fret", "0")
        return articulations

    def _track_bars(self, track: Track, articulations: dict[int, int]) -> list[int]:
        tuning = track.tuning or DEFAULT_TUNINGS[track.role]
        placed: list[Placed] = (
            place_percussion(track.notes, self.starts[-1])
            if track.is_percussion
            else place_fretted(track.notes, tuning, frets=24)
        )
        ids = []
        for bar in range(self.score.n_bars):
            bar_id = self._next("bar")
            voice_id = self._next("voice")
            bar_el = _sub(self.bars_el, "Bar", id=str(bar_id))
            _sub(bar_el, "Clef", "Neutral" if track.is_percussion else "G2")
            _sub(bar_el, "Voices", f"{voice_id} -1 -1 -1")
            beats = layout_bar(placed, self.starts[bar], self.starts[bar + 1])
            beat_ids = [self._beat(b, track, len(tuning), articulations) for b in beats]
            voice = _sub(self.voices_el, "Voice", id=str(voice_id))
            _sub(voice, "Beats", " ".join(str(i) for i in beat_ids))
            ids.append(bar_id)
        return ids

    def _beat(
        self, tab_beat: TabBeat, track: Track, n_strings: int, articulations: dict[int, int]
    ) -> int:
        beat_id = self._next("beat")
        beat = _sub(self.beats_el, "Beat", id=str(beat_id))
        if tab_beat.notes:
            plain = [
                n.placed.note.velocity
                for n in tab_beat.notes
                if not n.placed.note.flags & _LOUDNESS_FLAGS
            ]
            velocity = plain[0] if plain else min(n.placed.note.velocity for n in tab_beat.notes)
            _sub(beat, "Dynamic", nearest_dynamic(velocity))
        _sub(beat, "Rhythm", ref=str(self._rhythm(tab_beat.value)))
        if tab_beat.notes:
            note_ids = [self._note(n, track, n_strings, articulations) for n in tab_beat.notes]
            _sub(beat, "Notes", " ".join(str(i) for i in note_ids))
        return beat_id

    def _note(
        self, tab_note: TabNote, track: Track, n_strings: int, articulations: dict[int, int]
    ) -> int:
        note_id = self._next("note")
        placed, note = tab_note.placed, tab_note.placed.note
        element = _sub(self.notes_el, "Note", id=str(note_id))
        if tab_note.tie_in or tab_note.tie_out:
            _sub(
                element,
                "Tie",
                origin=str(tab_note.tie_out).lower(),
                destination=str(tab_note.tie_in).lower(),
            )
        if not tab_note.tie_in:
            accent = sum(bit for flag, bit in _ACCENT_BITS if note.flags & flag)
            if accent:
                _sub(element, "Accent", str(accent))
            if note.flags & NoteFlag.GHOST:
                _sub(element, "AntiAccent", "Normal")
        if track.is_percussion:
            _sub(element, "InstrumentArticulation", str(articulations[note.pitch]))
        properties = _sub(element, "Properties")
        if not track.is_percussion:
            _sub(
                _sub(properties, "Property", name="String"),
                "String",
                str(n_strings - placed.string),
            )
            _sub(_sub(properties, "Property", name="Fret"), "Fret", str(placed.fret))
        if note.flags & NoteFlag.PALM_MUTE:
            _sub(_sub(properties, "Property", name="PalmMuted"), "Enable")
        if note.flags & NoteFlag.DEAD and not tab_note.tie_in:
            _sub(_sub(properties, "Property", name="Muted"), "Enable")
        return note_id

    def _rhythm(self, value: NoteValue) -> int:
        key = (value.value, value.dotted, value.triplet)
        if key not in self.rhythm_ids:
            rhythm_id = len(self.rhythm_ids)
            self.rhythm_ids[key] = rhythm_id
            rhythm = _sub(self.rhythms_el, "Rhythm", id=str(rhythm_id))
            _sub(rhythm, "NoteValue", _NOTE_VALUE_NAMES[value.value])
            if value.dotted:
                _sub(rhythm, "AugmentationDot", count="1")
            if value.triplet:
                _sub(rhythm, "PrimaryTuplet", num="3", den="2")
        return self.rhythm_ids[key]
