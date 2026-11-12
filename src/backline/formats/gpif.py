# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Guitar Pro 7/8 ``.gp`` files: a ZIP container whose ``Content/score.gpif`` is XML.

The document is a set of id-linked tables (master bars -> bars per track -> voices ->
beats -> notes, with shared rhythms). The reader keeps every voice of the first staff of
each track, merges ties, drops grace notes, replays simile (bar-repeat) marks, applies the
triplet feel of each bar and resolves drum notes through the track's articulation list.
Repeats are not unrolled. Files come from the internet, so the XML is parsed with
:mod:`defusedxml` and the archive entry size is capped.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Final
from xml.etree.ElementTree import Element, ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import fromstring

from backline.formats.common import ReadStats, ScoreReadError, TrackNotes, guess_role
from backline.formats.tablature import GP_DYNAMICS
from backline.score import TPQ, Note, NoteFlag, Score, Tempo, TimeSignature, Track
from backline.timing import Feel, apply_feel

GPIF_ENTRY: Final = "Content/score.gpif"
MAX_GPIF_BYTES: Final = 64 * 1024 * 1024

_NOTE_VALUES: Final = {
    "Long": Fraction(16),
    "DoubleWhole": Fraction(8),
    "Whole": Fraction(4),
    "Half": Fraction(2),
    "Quarter": Fraction(1),
    "Eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
    "128th": Fraction(1, 32),
    "256th": Fraction(1, 64),
}
"""Note values in quarter notes."""

_TEMPO_UNITS: Final = {
    1: Fraction(1, 2),
    2: Fraction(1),
    3: Fraction(3, 2),
    4: Fraction(2),
    5: Fraction(3),
}
"""Beat unit of a tempo mark (8th, quarter, dotted quarter, half, dotted half) in quarters."""

_FEELS: Final = {
    "Triplet8th": Feel.TRIPLET_8,
    "Triplet16th": Feel.TRIPLET_16,
    "Dotted8th": Feel.DOTTED_8,
    "Dotted16th": Feel.DOTTED_16,
    "Scottish8th": Feel.SCOTTISH_8,
    "Scottish16th": Feel.SCOTTISH_16,
}

_SIMILE_SOURCE: Final = {"Simple": 1, "FirstOfDouble": 2, "SecondOfDouble": 2}
"""How many bars back a simile mark repeats."""

_DEFAULT_DYNAMIC: Final = "F"
_ACCENT_HEAVY: Final = 0x04
_ACCENT_NORMAL: Final = 0x08


@dataclass(slots=True)
class _TrackInfo:
    name: str
    instrument: str
    percussion: bool
    program: int
    tuning: tuple[int, ...] | None
    capo: int
    staves: int
    articulations: list[tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Event:
    """A note of one bar, relative to the bar start, before ties are merged."""

    offset: int
    duration: int
    pitch: int
    velocity: int
    flags: NoteFlag
    key: tuple[int, int]
    tie: bool


def read_gp(
    path: Path, *, stats: ReadStats | None = None, apply_triplet_feel: bool = True
) -> Score:
    """Read a Guitar Pro 7/8 ``.gp`` file."""
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(GPIF_ENTRY)
            if info.file_size > MAX_GPIF_BYTES:
                raise ScoreReadError(f"{path.name}: {GPIF_ENTRY} is too large")
            data = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ScoreReadError(f"{path.name} is not a Guitar Pro 7+ archive") from exc
    except KeyError as exc:
        raise ScoreReadError(f"{path.name} has no {GPIF_ENTRY}") from exc
    score = parse_gpif(data, stats=stats, apply_triplet_feel=apply_triplet_feel)
    if not score.title:
        score.title = path.stem
    return score


def parse_gpif(
    data: bytes, *, stats: ReadStats | None = None, apply_triplet_feel: bool = True
) -> Score:
    """Parse the ``score.gpif`` XML document."""
    counters = stats if stats is not None else ReadStats()
    try:
        root = fromstring(data)
    except (ParseError, DefusedXmlException) as exc:
        raise ScoreReadError(f"invalid GPIF document: {exc}") from exc
    return _Reader(root, counters).score(apply_triplet_feel=apply_triplet_feel)


def _text(element: Element | None, path: str, default: str = "") -> str:
    if element is None:
        return default
    node = element.find(path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except ValueError:
        return default


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split() if part.lstrip("-").isdigit()]


class _Reader:
    def __init__(self, root: Element, stats: ReadStats) -> None:
        self.root = root
        self.stats = stats
        self.bars = {b.get("id", ""): b for b in root.iterfind("Bars/Bar")}
        self.voices = {v.get("id", ""): v for v in root.iterfind("Voices/Voice")}
        self.beats = {b.get("id", ""): b for b in root.iterfind("Beats/Beat")}
        self.notes = {n.get("id", ""): n for n in root.iterfind("Notes/Note")}
        self.rhythms = {r.get("id", ""): self._rhythm(r) for r in root.iterfind("Rhythms/Rhythm")}

    # ----------------------------------------------------------------- structure

    def score(self, *, apply_triplet_feel: bool) -> Score:
        masters = self.root.findall("MasterBars/MasterBar")
        if not masters:
            raise ScoreReadError("the document has no bars")
        meters = [self._meter(_text(mb, "Time", "4/4")) for mb in masters]
        time_signatures: list[TimeSignature] = []
        starts = [0]
        for bar, (numerator, denominator) in enumerate(meters):
            if not time_signatures or (numerator, denominator) != (
                time_signatures[-1].numerator,
                time_signatures[-1].denominator,
            ):
                time_signatures.append(TimeSignature(bar, numerator, denominator))
            starts.append(starts[-1] + numerator * TPQ * 4 // denominator)

        infos = self._tracks()
        staff_offsets = []
        offset = 0
        for info in infos:
            staff_offsets.append(offset)
            offset += info.staves
            if info.staves > 1:
                self.stats["extra_staves_ignored"] += info.staves - 1

        bar_ids = [_ints(_text(mb, "Bars")) for mb in masters]
        tracks = []
        for info, staff in zip(infos, staff_offsets, strict=True):
            ids = [row[staff] if staff < len(row) else -1 for row in bar_ids]
            tracks.append(self._track(info, ids, starts))

        score = Score(
            n_bars=len(masters),
            time_signatures=time_signatures,
            tempos=self._tempos(starts),
            tracks=tracks,
            title=_text(self.root, "Score/Title"),
            artist=_text(self.root, "Score/Artist"),
        )
        if apply_triplet_feel:
            feels = {
                bar: _FEELS.get(_text(mb, "TripletFeel"), Feel.STRAIGHT)
                for bar, mb in enumerate(masters)
            }
            apply_feel(score, feels)
        return score

    @staticmethod
    def _meter(text: str) -> tuple[int, int]:
        numerator, _, denominator = text.partition("/")
        num, den = _int(numerator, 4), _int(denominator, 4)
        if num <= 0 or den <= 0 or den & (den - 1):
            raise ScoreReadError(f"unsupported time signature {text!r}")
        return num, den

    def _tracks(self) -> list[_TrackInfo]:
        elements = {t.get("id", ""): t for t in self.root.iterfind("Tracks/Track")}
        order = _ints(_text(self.root, "MasterTrack/Tracks"))
        ids = [str(i) for i in order if str(i) in elements] or list(elements)
        return [self._track_info(elements[i]) for i in ids]

    def _track_info(self, element: Element) -> _TrackInfo:
        instrument = _text(element, "InstrumentSet/Type")
        program_text = _text(element, "Sounds/Sound/MIDI/Program") or _text(
            element, "GeneralMidi/Program"
        )
        channel_text = _text(element, "MidiConnection/PrimaryChannel") or _text(
            element, "GeneralMidi/PrimaryChannel"
        )
        general = element.find("GeneralMidi")
        percussion = (
            instrument == "drumKit"
            or _int(channel_text, -1) == 9
            or (general is not None and general.get("table") == "Percussion")
        )
        staves = element.findall("Staves/Staff")
        staff = staves[0] if staves else element
        tuning_text = _text(staff, "Properties/Property[@name='Tuning']/Pitches") or _text(
            element, "Properties/Property[@name='Tuning']/Pitches"
        )
        capo_text = _text(staff, "Properties/Property[@name='CapoFret']/Fret") or _text(
            element, "Properties/Property[@name='CapoFret']/Fret"
        )
        tuning = tuple(_ints(tuning_text)) or None
        info = _TrackInfo(
            name=_text(element, "Name"),
            instrument=instrument,
            percussion=percussion,
            program=_int(program_text, 0),
            tuning=None if percussion else tuning,
            capo=0 if percussion else _int(capo_text, 0),
            staves=max(len(staves), 1),
        )
        for drum in element.iterfind("InstrumentSet/Elements/Element"):
            element_name = _text(drum, "Name")
            for articulation in drum.iterfind("Articulations/Articulation"):
                midi = _int(_text(articulation, "OutputMidiNumber"), -1)
                label = _text(articulation, "Name")
                info.articulations.append((midi, f"{element_name}.{label}".strip(".")))
        return info

    def _tempos(self, starts: list[int]) -> list[Tempo]:
        tempos: dict[int, float] = {}
        for automation in self.root.iterfind("MasterTrack/Automations/Automation"):
            if _text(automation, "Type") != "Tempo":
                continue
            parts = _text(automation, "Value").split()
            if not parts:
                continue
            try:
                value = Fraction(parts[0])
            except ValueError:
                continue
            reference = _int(parts[1], 2) if len(parts) > 1 else 1
            unit = _TEMPO_UNITS.get(reference, Fraction(1))
            bar = min(max(_int(_text(automation, "Bar"), 0), 0), len(starts) - 2)
            try:
                position = Fraction(_text(automation, "Position", "0"))
            except ValueError:
                position = Fraction(0)
            tick = starts[bar] + round((starts[bar + 1] - starts[bar]) * position)
            bpm = float(value * unit)
            if bpm > 0:
                tempos[tick] = round(bpm, 3)
        if 0 not in tempos:
            tempos[0] = tempos[min(tempos)] if tempos else 120.0
        return [Tempo(tick, bpm) for tick, bpm in sorted(tempos.items())]

    # ----------------------------------------------------------------- notes

    @staticmethod
    def _rhythm(element: Element) -> Fraction:
        value = _NOTE_VALUES.get(_text(element, "NoteValue"), Fraction(1))
        dot = element.find("AugmentationDot")
        if dot is not None:
            dots = _int(dot.get("count", "0"), 0)
            value *= 2 - Fraction(1, 2**dots)
        tuplet = element.find("PrimaryTuplet")
        if tuplet is not None:
            num, den = _int(tuplet.get("num", "1"), 1), _int(tuplet.get("den", "1"), 1)
            if num > 0 and den > 0:
                value *= Fraction(den, num)
        return value * TPQ

    def _track(self, info: _TrackInfo, bar_ids: list[int], starts: list[int]) -> Track:
        per_bar: list[list[_Event]] = []
        collected = TrackNotes()
        names: dict[int, str] = {}
        for bar, bar_id in enumerate(bar_ids):
            bar_element = self.bars.get(str(bar_id))
            simile = _text(bar_element, "SimileMark")
            if simile in _SIMILE_SOURCE and bar - _SIMILE_SOURCE[simile] >= 0:
                events = per_bar[bar - _SIMILE_SOURCE[simile]]
                self.stats["simile_bars_expanded"] += 1
            else:
                events = self._bar_events(info, bar_element, names)
            per_bar.append(events)
            for event in events:
                start = starts[bar] + event.offset
                end = start + event.duration
                if event.tie:
                    if not collected.extend(event.key, start, end):
                        self.stats["orphan_ties"] += 1
                    continue
                collected.add(
                    event.key, Note(start, event.pitch, event.duration, event.velocity, event.flags)
                )
        return Track(
            name=info.name,
            role=guess_role(info.program, info.percussion),
            notes=sorted(collected.notes),
            program=info.program,
            is_percussion=info.percussion,
            tuning=info.tuning,
            instrument=info.instrument,
            percussion_names=names,
        )

    def _bar_events(
        self, info: _TrackInfo, bar: Element | None, names: dict[int, str]
    ) -> list[_Event]:
        if bar is None:
            return []
        events: list[_Event] = []
        # Ties continue within the same voice slot of the next bar, so the slot (not the
        # per-bar voice id) identifies a voice.
        for slot, voice_id in enumerate(_ints(_text(bar, "Voices"))):
            voice = self.voices.get(str(voice_id)) if voice_id >= 0 else None
            if voice is None:
                continue
            position = Fraction(0)
            for beat_id in _ints(_text(voice, "Beats")):
                beat = self.beats.get(str(beat_id))
                if beat is None:
                    continue
                note_ids = _ints(_text(beat, "Notes"))
                if _text(beat, "GraceNotes"):
                    self.stats["grace_notes_dropped"] += len(note_ids)
                    continue
                rhythm = beat.find("Rhythm")
                duration = self.rhythms.get(rhythm.get("ref", "") if rhythm is not None else "")
                if duration is None:
                    self.stats["beats_without_rhythm"] += 1
                    continue
                velocity = GP_DYNAMICS.get(_text(beat, "Dynamic"), GP_DYNAMICS[_DEFAULT_DYNAMIC])
                offset = round(position)
                length = max(round(position + duration) - offset, 1)
                for note_id in note_ids:
                    event = self._note(info, note_id, offset, length, velocity, slot, names)
                    if event is not None:
                        events.append(event)
                position += duration
        return events

    def _note(
        self,
        info: _TrackInfo,
        note_id: int,
        offset: int,
        length: int,
        velocity: int,
        slot: int,
        names: dict[int, str],
    ) -> _Event | None:
        note = self.notes.get(str(note_id))
        if note is None:
            return None
        properties = {p.get("name", ""): p for p in note.iterfind("Properties/Property")}
        if info.percussion:
            pitch = self._drum_pitch(info, note, properties, names)
            key = (slot, -1 if pitch is None else pitch)
        else:
            pitch, string = self._fretted_pitch(info, properties)
            key = (slot, string)
        if pitch is None:
            return None
        if not 0 <= pitch <= 127:
            self.stats["notes_out_of_midi_range"] += 1
            return None

        flags = NoteFlag.NONE
        accent = _int(_text(note, "Accent"), 0)
        if accent & _ACCENT_HEAVY:
            flags |= NoteFlag.HEAVY_ACCENT
        if accent & _ACCENT_NORMAL:
            flags |= NoteFlag.ACCENT
        if _text(note, "AntiAccent").lower() == "normal":
            flags |= NoteFlag.GHOST
        if "PalmMuted" in properties and properties["PalmMuted"].find("Enable") is not None:
            flags |= NoteFlag.PALM_MUTE
        if "Muted" in properties and properties["Muted"].find("Enable") is not None:
            flags |= NoteFlag.DEAD
        tie = note.find("Tie")
        tied = tie is not None and tie.get("destination", "").lower() == "true"
        return _Event(offset, length, pitch, velocity, flags, key, tied)

    def _fretted_pitch(
        self, info: _TrackInfo, properties: dict[str, Element]
    ) -> tuple[int | None, int]:
        if "String" in properties and "Fret" in properties and info.tuning:
            string = _int(_text(properties["String"], "String"), -1)
            fret = _int(_text(properties["Fret"], "Fret"), -1)
            if 0 <= string < len(info.tuning) and fret >= 0:
                return info.tuning[string] + fret + info.capo, string
            self.stats["notes_with_bad_string"] += 1
            return None, -1
        if "Midi" in properties:
            return _int(_text(properties["Midi"], "Number"), -1), -1
        if "Tone" in properties and "Octave" in properties:
            tone = _int(_text(properties["Tone"], "Step"), 0)
            octave = _int(_text(properties["Octave"], "Number"), 0)
            return octave * 12 + tone, -1
        self.stats["notes_without_pitch"] += 1
        return None, -1

    def _drum_pitch(
        self,
        info: _TrackInfo,
        note: Element,
        properties: dict[str, Element],
        names: dict[int, str],
    ) -> int | None:
        index_text = _text(note, "InstrumentArticulation")
        if index_text:
            index = _int(index_text, -1)
            if 0 <= index < len(info.articulations):
                midi, label = info.articulations[index]
                if midi >= 0:
                    names.setdefault(midi, label)
                    return midi
            self.stats["unknown_drum_articulations"] += 1
            return None
        if "Midi" in properties:
            return _int(_text(properties["Midi"], "Number"), -1)
        if "Element" in properties:
            self.stats["gp6_drum_notes_skipped"] += 1
            return None
        self.stats["notes_without_pitch"] += 1
        return None
