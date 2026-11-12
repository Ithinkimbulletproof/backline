# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""GPIF (Guitar Pro 7/8) reader: synthetic round trips plus hand-written documents."""

import zipfile
from pathlib import Path

import pytest

from backline.config import BacklineConfig
from backline.formats.common import ReadStats, ScoreReadError
from backline.formats.gpif import GPIF_ENTRY, parse_gpif, read_gp
from backline.formats.gpif_writer import write_gp
from backline.score import TPQ, NoteFlag, Tempo
from backline.synth import SynthSong
from tests.support import note_signature

BAR = 4 * TPQ

RHYTHMS = """
<Rhythm id="q"><NoteValue>Quarter</NoteValue></Rhythm>
<Rhythm id="e"><NoteValue>Eighth</NoteValue></Rhythm>
<Rhythm id="t"><NoteValue>Eighth</NoteValue><PrimaryTuplet num="3" den="2"/></Rhythm>
<Rhythm id="dd"><NoteValue>Quarter</NoteValue><AugmentationDot count="2"/></Rhythm>
<Rhythm id="s"><NoteValue>16th</NoteValue></Rhythm>
<Rhythm id="w"><NoteValue>Whole</NoteValue></Rhythm>
"""

GUITAR = """
<Track id="0">
  <Name><![CDATA[Lead]]></Name>
  <InstrumentSet><Type>electricGuitar</Type></InstrumentSet>
  <Sounds><Sound><MIDI><Program>29</Program></MIDI></Sound></Sounds>
  <MidiConnection><PrimaryChannel>0</PrimaryChannel></MidiConnection>
  <Staves><Staff><Properties>
    <Property name="Tuning"><Pitches>40 45 50 55 59 64</Pitches></Property>{capo}
  </Properties></Staff></Staves>
</Track>
"""

DRUMS = """
<Track id="1">
  <Name><![CDATA[Kit]]></Name>
  <InstrumentSet>
    <Type>drumKit</Type>
    <Elements>
      <Element><Name>Kick</Name><Articulations>
        <Articulation><Name>Hit</Name><OutputMidiNumber>36</OutputMidiNumber></Articulation>
      </Articulations></Element>
      <Element><Name>Snare</Name><Articulations>
        <Articulation><Name>Hit</Name><OutputMidiNumber>38</OutputMidiNumber></Articulation>
        <Articulation><Name>Side Stick</Name><OutputMidiNumber>37</OutputMidiNumber></Articulation>
      </Articulations></Element>
    </Elements>
  </InstrumentSet>
  <MidiConnection><PrimaryChannel>9</PrimaryChannel></MidiConnection>
  <Staves><Staff><Properties/></Staff></Staves>
</Track>
"""


def _doc(
    *,
    tracks: str = GUITAR.format(capo=""),
    track_order: str = "0",
    masterbars: str,
    bars: str,
    voices: str,
    beats: str,
    notes: str = "",
    automations: str = "",
) -> bytes:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<GPIF>
  <Score><Title><![CDATA[Hand Made]]></Title><Artist><![CDATA[Nobody]]></Artist></Score>
  <MasterTrack><Tracks>{track_order}</Tracks><Automations>{automations}</Automations></MasterTrack>
  <Tracks>{tracks}</Tracks>
  <MasterBars>{masterbars}</MasterBars>
  <Bars>{bars}</Bars>
  <Voices>{voices}</Voices>
  <Beats>{beats}</Beats>
  <Notes>{notes}</Notes>
  <Rhythms>{RHYTHMS}</Rhythms>
</GPIF>""".encode()


def _master(bars: str, time: str = "4/4", extra: str = "") -> str:
    return f"<MasterBar><Time>{time}</Time><Bars>{bars}</Bars>{extra}</MasterBar>"


def _note(note_id: str, string: int, fret: int, extra: str = "", props: str = "") -> str:
    return (
        f'<Note id="{note_id}">{extra}<Properties>'
        f'<Property name="String"><String>{string}</String></Property>'
        f'<Property name="Fret"><Fret>{fret}</Fret></Property>{props}'
        "</Properties></Note>"
    )


def _beat(beat_id: str, rhythm: str, notes: str = "", extra: str = "") -> str:
    notes_xml = f"<Notes>{notes}</Notes>" if notes else ""
    return f'<Beat id="{beat_id}">{extra}<Rhythm ref="{rhythm}"/>{notes_xml}</Beat>'


# ------------------------------------------------------------------ round trip


def test_round_trip_of_synthetic_songs(
    corpus: list[SynthSong], tiny_config: BacklineConfig, tmp_path: Path
) -> None:
    for index, song in enumerate(corpus[:10]):
        path = tmp_path / f"{index}.gp"
        write_gp(song.score, path)
        stats = ReadStats()
        back = read_gp(path, stats=stats)
        assert note_signature(back, tiny_config.velocity) == note_signature(
            song.score, tiny_config.velocity
        )
        assert (back.title, back.artist) == (song.score.title, song.score.artist)
        assert back.time_signatures == song.score.time_signatures
        assert back.tempos == song.score.tempos
        assert [t.tuning for t in back.tracks] == [t.tuning for t in song.score.tracks]
        assert [t.role for t in back.tracks] == [t.role for t in song.score.tracks]
        drums = next(t for t in back.tracks if t.is_percussion)
        assert drums.instrument == "drumKit"
        assert set(drums.percussion_names) == {n.pitch for n in drums.notes}
        assert stats == ReadStats()


# -------------------------------------------------------------- hand-made docs


def test_rhythms_strings_and_dynamics() -> None:
    doc = _doc(
        masterbars=_master("0"),
        bars='<Bar id="0"><Voices>0 -1 -1 -1</Voices></Bar>',
        voices='<Voice id="0"><Beats>0 1 2 3 4 5</Beats></Voice>',
        beats="".join(
            [
                _beat("0", "q", "0", "<Dynamic>MF</Dynamic>"),
                _beat("1", "t", "1"),
                _beat("2", "t", "2"),
                _beat("3", "t", "3"),
                _beat("4", "dd", "4", "<Dynamic>FFF</Dynamic>"),
                _beat("5", "s"),
            ]
        ),
        notes="".join(
            [
                _note("0", 0, 0),
                _note("1", 1, 0),
                _note("2", 1, 2),
                _note("3", 2, 0),
                _note("4", 5, 3),
            ]
        ),
    )
    score = parse_gpif(doc)
    (track,) = score.tracks
    assert track.name == "Lead"
    assert track.instrument == "electricGuitar"
    assert track.tuning == (40, 45, 50, 55, 59, 64)
    assert [(n.start, n.duration, n.pitch) for n in track.notes] == [
        (0, 960, 40),
        (960, 320, 45),
        (1280, 320, 47),
        (1600, 320, 50),
        (1920, 1680, 67),
    ]
    assert [n.velocity for n in track.notes] == [79, 95, 95, 95, 127]
    assert (score.title, score.artist) == ("Hand Made", "Nobody")


def test_tempo_marks_use_their_beat_unit() -> None:
    doc = _doc(
        masterbars=_master("0") + _master("1"),
        bars='<Bar id="0"><Voices>-1 -1 -1 -1</Voices></Bar><Bar id="1"><Voices>-1</Voices></Bar>',
        voices="",
        beats="",
        automations=(
            "<Automation><Type>Tempo</Type><Bar>0</Bar><Position>0</Position>"
            "<Value>120 3</Value></Automation>"
            "<Automation><Type>Tempo</Type><Bar>1</Bar><Position>0.5</Position>"
            "<Value>100 1</Value></Automation>"
        ),
    )
    assert parse_gpif(doc).tempos == [Tempo(0, 180.0), Tempo(BAR + BAR // 2, 50.0)]


def test_ties_merge_across_bars() -> None:
    doc = _doc(
        masterbars=_master("0") + _master("1"),
        bars='<Bar id="0"><Voices>0</Voices></Bar><Bar id="1"><Voices>1</Voices></Bar>',
        voices='<Voice id="0"><Beats>0</Beats></Voice><Voice id="1"><Beats>1</Beats></Voice>',
        beats=_beat("0", "w", "0") + _beat("1", "w", "1"),
        notes=_note("0", 0, 5, '<Tie origin="true" destination="false"/>')
        + _note("1", 0, 5, '<Tie origin="false" destination="true"/>'),
    )
    (note,) = parse_gpif(doc).tracks[0].notes
    assert (note.start, note.duration, note.pitch) == (0, 2 * BAR, 45)


def test_grace_notes_dropped_and_voices_merged() -> None:
    stats = ReadStats()
    doc = _doc(
        masterbars=_master("0"),
        bars='<Bar id="0"><Voices>0 1 -1 -1</Voices></Bar>',
        voices='<Voice id="0"><Beats>0 1</Beats></Voice><Voice id="1"><Beats>2</Beats></Voice>',
        beats=_beat("0", "s", "0", "<GraceNotes>BeforeBeat</GraceNotes>")
        + _beat("1", "w", "1")
        + _beat("2", "w", "2"),
        notes=_note("0", 0, 2) + _note("1", 0, 0) + _note("2", 5, 0),
    )
    notes = parse_gpif(doc, stats=stats).tracks[0].notes
    assert [(n.start, n.pitch) for n in notes] == [(0, 40), (0, 64)]
    assert stats["grace_notes_dropped"] == 1


def test_simile_mark_repeats_the_previous_bar() -> None:
    stats = ReadStats()
    doc = _doc(
        masterbars=_master("0") + _master("1"),
        bars='<Bar id="0"><Voices>0</Voices></Bar>'
        '<Bar id="1"><SimileMark>Simple</SimileMark><Voices>-1</Voices></Bar>',
        voices='<Voice id="0"><Beats>0 1</Beats></Voice>',
        beats=_beat("0", "q", "0") + _beat("1", "dd", "1"),
        notes=_note("0", 0, 0) + _note("1", 0, 3),
    )
    notes = parse_gpif(doc, stats=stats).tracks[0].notes
    assert [(n.start, n.pitch) for n in notes] == [(0, 40), (960, 43), (BAR, 40), (BAR + 960, 43)]
    assert stats["simile_bars_expanded"] == 1


def test_drum_articulations_resolve_to_midi_notes() -> None:
    stats = ReadStats()
    doc = _doc(
        tracks=DRUMS,
        track_order="1",
        masterbars=_master("0"),
        bars='<Bar id="0"><Voices>0</Voices></Bar>',
        voices='<Voice id="0"><Beats>0 1</Beats></Voice>',
        beats=_beat("0", "q", "0 1") + _beat("1", "dd", "2"),
        notes='<Note id="0"><InstrumentArticulation>0</InstrumentArticulation></Note>'
        '<Note id="1"><AntiAccent>Normal</AntiAccent>'
        "<InstrumentArticulation>2</InstrumentArticulation></Note>"
        '<Note id="2"><InstrumentArticulation>9</InstrumentArticulation></Note>',
    )
    (kit,) = parse_gpif(doc, stats=stats).tracks
    assert kit.is_percussion
    assert kit.tuning is None
    assert [(n.pitch, n.flags) for n in kit.notes] == [(36, NoteFlag.NONE), (37, NoteFlag.GHOST)]
    assert kit.percussion_names == {36: "Kick.Hit", 37: "Snare.Side Stick"}
    assert stats["unknown_drum_articulations"] == 1


def test_capo_and_articulation_flags() -> None:
    capo = '<Property name="CapoFret"><Fret>2</Fret></Property>'
    doc = _doc(
        tracks=GUITAR.format(capo=capo),
        masterbars=_master("0"),
        bars='<Bar id="0"><Voices>0</Voices></Bar>',
        voices='<Voice id="0"><Beats>0 1 2 3</Beats></Voice>',
        beats=_beat("0", "q", "0")
        + _beat("1", "q", "1")
        + _beat("2", "q", "2")
        + _beat("3", "q", "3"),
        notes=_note("0", 0, 0, "<Accent>4</Accent>")
        + _note("1", 0, 0, "<Accent>8</Accent>")
        + _note("2", 0, 0, props='<Property name="PalmMuted"><Enable/></Property>')
        + _note("3", 0, 0, props='<Property name="Muted"><Enable/></Property>'),
    )
    notes = parse_gpif(doc).tracks[0].notes
    assert {n.pitch for n in notes} == {42}
    assert [n.flags for n in notes] == [
        NoteFlag.HEAVY_ACCENT,
        NoteFlag.ACCENT,
        NoteFlag.PALM_MUTE,
        NoteFlag.DEAD,
    ]


def test_triplet_feel_mark() -> None:
    doc = _doc(
        masterbars=_master("0", extra="<TripletFeel>Triplet8th</TripletFeel>"),
        bars='<Bar id="0"><Voices>0</Voices></Bar>',
        voices='<Voice id="0"><Beats>0 1 2 3 4 5 6 7</Beats></Voice>',
        beats="".join(_beat(str(i), "e", str(i)) for i in range(8)),
        notes="".join(_note(str(i), 0, 0) for i in range(8)),
    )
    assert [n.start for n in parse_gpif(doc).tracks[0].notes[:4]] == [0, 640, 960, 1600]
    straight = parse_gpif(doc, apply_triplet_feel=False).tracks[0].notes
    assert [n.start for n in straight[:2]] == [0, 480]


def test_only_the_first_staff_of_a_track_is_read() -> None:
    two_staves = GUITAR.format(capo="").replace(
        "</Staves>", "<Staff><Properties/></Staff></Staves>"
    )
    stats = ReadStats()
    doc = _doc(
        tracks=two_staves + DRUMS,
        track_order="0 1",
        masterbars=_master("0 1 2"),
        bars='<Bar id="0"><Voices>0</Voices></Bar><Bar id="1"><Voices>1</Voices></Bar>'
        '<Bar id="2"><Voices>2</Voices></Bar>',
        voices='<Voice id="0"><Beats>0</Beats></Voice><Voice id="1"><Beats>1</Beats></Voice>'
        '<Voice id="2"><Beats>2</Beats></Voice>',
        beats=_beat("0", "w", "0") + _beat("1", "w", "1") + _beat("2", "w", "2"),
        notes=_note("0", 0, 0)
        + _note("1", 0, 7)
        + '<Note id="2"><InstrumentArticulation>1</InstrumentArticulation></Note>',
    )
    guitar, kit = parse_gpif(doc, stats=stats).tracks
    assert [n.pitch for n in guitar.notes] == [40]
    assert [n.pitch for n in kit.notes] == [38]
    assert stats["extra_staves_ignored"] == 1


def test_meter_changes_build_the_bar_grid() -> None:
    doc = _doc(
        masterbars=_master("0") + _master("1", time="7/8"),
        bars='<Bar id="0"><Voices>-1</Voices></Bar><Bar id="1"><Voices>-1</Voices></Bar>',
        voices="",
        beats="",
    )
    score = parse_gpif(doc)
    assert [str(ts) for ts in score.time_signatures] == ["4/4", "7/8"]
    assert score.end == BAR + 7 * TPQ // 2


# ------------------------------------------------------------------- failures


def test_entity_expansion_is_rejected() -> None:
    bomb = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>
<GPIF><Score><Title>&lol2;</Title></Score></GPIF>"""
    with pytest.raises(ScoreReadError, match="invalid GPIF"):
        parse_gpif(bomb)


def test_not_a_zip(tmp_path: Path) -> None:
    path = tmp_path / "fake.gp"
    path.write_bytes(b"not a zip at all")
    with pytest.raises(ScoreReadError, match="not a Guitar Pro 7"):
        read_gp(path)


def test_zip_without_score(tmp_path: Path) -> None:
    path = tmp_path / "empty.gp"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("VERSION", "7.0")
    with pytest.raises(ScoreReadError, match=GPIF_ENTRY):
        read_gp(path)


def test_document_without_bars() -> None:
    with pytest.raises(ScoreReadError, match="no bars"):
        parse_gpif(b"<GPIF><MasterBars/></GPIF>")
