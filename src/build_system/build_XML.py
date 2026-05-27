import sys
import re
import math
import enum
import dataclasses
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union, Any
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement

import numpy as np
from numpy import ndarray

from src.core import layers
from src.extraction.symbol_extraction import Barline, Clef, Accidental, Rest, AccidentalType, ClefType, RestType
from src.extraction.notegroup_extraction import NoteGroup
from src.extraction.notehead_extraction import NoteHead, NoteType
from src.utils.utils import get_global_unit_size, get_total_track_nums
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
DIVISION_PER_QUARTER = 16

G_CLEF_POS_TO_PITCH = ['D', 'E', 'F', 'G', 'A', 'B', 'C']
F_CLEF_POS_TO_PITCH = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
C_CLEF_POS_TO_PITCH = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

SHARP_KEY_ORDER = ['F', 'C', 'G', 'D', 'A', 'E']
FLAT_KEY_ORDER  = ['B', 'E', 'A', 'D', 'G', 'C']

NOTE_TYPE_TO_RHYTHM: Dict[NoteType, Dict[str, Any]] = {
    NoteType.WHOLE: {'name': 'whole', 'duration': DIVISION_PER_QUARTER * 4},
    NoteType.HALF: {'name': 'half', 'duration': DIVISION_PER_QUARTER * 2},
    NoteType.HALF_OR_WHOLE: {'name': 'half', 'duration': DIVISION_PER_QUARTER * 2},
    NoteType.QUARTER: {'name': 'quarter', 'duration': DIVISION_PER_QUARTER},
    NoteType.EIGHTH: {'name': 'eighth', 'duration': DIVISION_PER_QUARTER // 2},
    NoteType.SIXTEENTH: {'name': '16th', 'duration': DIVISION_PER_QUARTER // 4},
    NoteType.THIRTY_SECOND: {'name': '32nd', 'duration': DIVISION_PER_QUARTER // 8},
    NoteType.SIXTY_FOURTH: {'name': '64th', 'duration': DIVISION_PER_QUARTER // 16},
}

REST_TYPE_TO_DURATION: Dict[RestType, int] = {
    RestType.WHOLE: DIVISION_PER_QUARTER * 4,
    RestType.WHOLE_HALF: DIVISION_PER_QUARTER * 2,
    RestType.HALF: DIVISION_PER_QUARTER * 2,
    RestType.QUARTER: DIVISION_PER_QUARTER,
    RestType.EIGHTH: DIVISION_PER_QUARTER // 2,
    RestType.SIXTEENTH: DIVISION_PER_QUARTER // 4,
    RestType.THIRTY_SECOND: DIVISION_PER_QUARTER // 8,
    RestType.SIXTY_FOURTH: DIVISION_PER_QUARTER // 16,
}

REST_TYPE_TO_NAME: Dict[RestType, str] = {
    RestType.WHOLE: 'whole',
    RestType.WHOLE_HALF: 'half',
    RestType.HALF: 'half',
    RestType.QUARTER: 'quarter',
    RestType.EIGHTH: 'eighth',
    RestType.SIXTEENTH: '16th',
    RestType.THIRTY_SECOND: '32nd',
    RestType.SIXTY_FOURTH: '64th',
}

ACCIDENTAL_TO_ALTER: Dict[AccidentalType, str] = {
    AccidentalType.SHARP: '1',
    AccidentalType.FLAT: '-1',
    AccidentalType.NATURAL: '0',
}

ACCIDENTAL_TO_NAME: Dict[AccidentalType, str] = {
    AccidentalType.SHARP: 'sharp',
    AccidentalType.FLAT: 'flat',
    AccidentalType.NATURAL: 'natural',
}

_CLEF_PITCH_TABLE: Dict[ClefType, List[str]] = {
    ClefType.GCLEF: G_CLEF_POS_TO_PITCH,
    ClefType.FCLEF: F_CLEF_POS_TO_PITCH,
    ClefType.CCLEF: C_CLEF_POS_TO_PITCH,
}

_CLEF_OCT_OFFSET: Dict[ClefType, int] = {
    ClefType.GCLEF: 4,
    ClefType.FCLEF: 2,
    ClefType.CCLEF: 3,
}

_CLEF_PITCH_OFFSET: Dict[ClefType, int] = {
    ClefType.GCLEF: 1,
    ClefType.FCLEF: 3,
    ClefType.CCLEF: 0,
}

_CLEF_SIGN: Dict[ClefType, str] = {
    ClefType.GCLEF: 'G',
    ClefType.FCLEF: 'F',
    ClefType.CCLEF: 'C',
}

_CLEF_LINE: Dict[ClefType, str] = {
    ClefType.GCLEF: '2',
    ClefType.FCLEF: '4',
    ClefType.CCLEF: '3',
}


# InstrumentConfig
@dataclasses.dataclass
class InstrumentConfig:
    """Mô tả đầy đủ một nhạc cụ để sinh MusicXML đúng chuẩn.
    Attributes:
        name           : tên hiển thị, vd. "Violin"
        sound          : MusicXML instrument-sound, vd. "strings.violin"
        midi_program   : MIDI program number (1-indexed)
        midi_channel   : MIDI channel (1-16)
        num_staves     : số dòng kẻ — 1 hầu hết nhạc cụ, 2 piano/harp
        default_clefs  : clef mặc định cho mỗi staff, len phải == num_staves
        part_id        : XML part id
        instrument_id  : XML instrument id"""
    name: str
    sound: str
    midi_program: int
    midi_channel: int = 1
    num_staves:  int = 1
    default_clefs: List[ClefType] = dataclasses.field(default_factory=lambda: [ClefType.GCLEF])
    part_id: str = 'P1'
    instrument_id: str = 'P1-I1'

    def __post_init__(self) -> None:
        if len(self.default_clefs) != self.num_staves:
            raise ValueError(
                f"InstrumentConfig '{self.name}': "
                f"default_clefs có {len(self.default_clefs)} phần tử "
                f"nhưng num_staves={self.num_staves}"
            )


# Preset nhạc cụ
INSTRUMENT_PRESETS: Dict[str, InstrumentConfig] = {
    'piano': InstrumentConfig(
        name='Piano', sound='keyboard.piano', midi_program=1,
        num_staves=2, default_clefs=[ClefType.GCLEF, ClefType.FCLEF]),
    'violin': InstrumentConfig(
        name='Violin', sound='strings.violin', midi_program=41,
        default_clefs=[ClefType.GCLEF]),
    'viola': InstrumentConfig(
        name='Viola', sound='strings.viola', midi_program=42,
        default_clefs=[ClefType.CCLEF]),
    'cello': InstrumentConfig(
        name='Cello', sound='strings.cello', midi_program=43,
        default_clefs=[ClefType.FCLEF]),
    'double_bass': InstrumentConfig(
        name='Double Bass', sound='strings.contrabass', midi_program=44,
        default_clefs=[ClefType.FCLEF]),
    'flute': InstrumentConfig(
        name='Flute', sound='wind.flutes.flute', midi_program=74,
        default_clefs=[ClefType.GCLEF]),
    'oboe': InstrumentConfig(
        name='Oboe', sound='wind.reed.oboe', midi_program=69,
        default_clefs=[ClefType.GCLEF]),
    'clarinet': InstrumentConfig(
        name='Clarinet', sound='wind.reed.clarinet', midi_program=72,
        default_clefs=[ClefType.GCLEF]),
    'bassoon': InstrumentConfig(
        name='Bassoon', sound='wind.reed.bassoon', midi_program=71,
        default_clefs=[ClefType.FCLEF]),
    'trumpet': InstrumentConfig(
        name='Trumpet', sound='brass.trumpet', midi_program=57,
        default_clefs=[ClefType.GCLEF]),
    'horn': InstrumentConfig(
        name='Horn', sound='brass.french-horn', midi_program=61,
        default_clefs=[ClefType.FCLEF]),
    'trombone': InstrumentConfig(
        name='Trombone', sound='brass.trombone', midi_program=58,
        default_clefs=[ClefType.FCLEF]),
    'tuba': InstrumentConfig(
        name='Tuba', sound='brass.tuba', midi_program=59,
        default_clefs=[ClefType.FCLEF]),
    'guitar': InstrumentConfig(
        name='Guitar', sound='pluck.guitar.nylon-string', midi_program=25,
        default_clefs=[ClefType.GCLEF]),
    'harp': InstrumentConfig(
        name='Harp', sound='pluck.harp', midi_program=47,
        num_staves=2, default_clefs=[ClefType.GCLEF, ClefType.FCLEF]),
    'soprano_voice': InstrumentConfig(
        name='Soprano', sound='voice.soprano', midi_program=53,
        default_clefs=[ClefType.GCLEF]),
    'tenor_voice': InstrumentConfig(
        name='Tenor', sound='voice.tenor', midi_program=53,
        default_clefs=[ClefType.GCLEF]),
    'bass_voice': InstrumentConfig(
        name='Bass', sound='voice.bass', midi_program=53,
        default_clefs=[ClefType.FCLEF]),
}


# Key
class Key(enum.Enum):
    C_MAJOR = 0
    G_MAJOR = 1
    D_MAJOR = 2
    A_MAJOR = 3
    E_MAJOR = 4
    B_MAJOR = 5
    F_SHARP_MAJOR = 6
    F_MAJOR = -1
    B_FLAT_MAJOR = -2
    E_FLAT_MAJOR = -3
    A_FLAT_MAJOR = -4
    D_FLAT_MAJOR = -5
    G_FLAT_MAJOR = -6


# ActionContext
class ActionContext:
    def __init__(self, instrument: InstrumentConfig) -> None:
        self.instrument = instrument
        self.key: Optional[Key] = None
        self.clefs: List[Optional[Clef]] = [None] * instrument.num_staves
        self.accidental_state: Dict[str, Optional[AccidentalType]] = {
            chr(ord('A') + i): None for i in range(7)
        }

    def reset_accidentals(self) -> None:
        self.accidental_state = {chr(ord('A') + i): None for i in range(7)}
        if self.key is None:
            return
        if self.key.value > 0:
            for name in SHARP_KEY_ORDER[:self.key.value]:
                self.accidental_state[name] = AccidentalType.SHARP
        elif self.key.value < 0:
            for name in FLAT_KEY_ORDER[:-self.key.value]:
                self.accidental_state[name] = AccidentalType.FLAT

    def effective_clef(self, track: int) -> ClefType:
        """Clef hiện tại của track, fallback về default của nhạc cụ."""
        if self.clefs[track] is not None:
            return self.clefs[track].label
        return self.instrument.default_clefs[track]


# Voice
class Voice:
    def __init__(self) -> None:
        self.id: Optional[int] = None
        self.note_ids: List[int] = []
        self.stem_up: Optional[bool] = None
        self.group_id: Optional[int] = None
        self.x_center: Optional[float] = None
        self.label: Optional[NoteType] = None
        self.has_dot: bool = False
        self.group: Optional[int] = None
        self.track: Optional[int] = None
        self.duration: Optional[int] = None
        self.rhythm_name: Optional[str] = None

    def init(self) -> None:
        notes = layers.get_layer('notes')
        labels = [notes[nid].label for nid in self.note_ids]
        count: Dict[NoteType, int] = {}
        for lb in labels:
            count[lb] = count.get(lb, 0) + 1
        self.label = max(count, key=lambda k: count[k])

        n_dot = sum(1 for nid in self.note_ids if notes[nid].has_dot)
        self.has_dot = n_dot > len(self.note_ids) // 2

        for nid in self.note_ids:
            if notes[nid].label != self.label:
                notes[nid].force_set_label(self.label)
            notes[nid].has_dot = self.has_dot

        rhythm = NOTE_TYPE_TO_RHYTHM[self.label]
        self.rhythm_name = rhythm['name']
        self.duration    = rhythm['duration']
        if self.has_dot:
            self.duration = round(self.duration * 1.5)

    def __repr__(self) -> str:
        return (
            f"Voice {self.id}  group={self.group}  track={self.track}"
            f" rhythm={self.rhythm_name}  dur={self.duration}"
        )


# Measure
class Measure:
    def __init__(self) -> None:
        self.symbols: List[Any] = []
        self.double_barline: Optional[bool] = None
        self.has_clef: bool = False
        self.clefs: List[Clef] = []
        self.voices: List[Voice] = []
        self.accidentals: List[Accidental] = []
        self.rests: List[Rest] = []
        self.number: Optional[int] = None
        self.at_beginning: Optional[bool] = None
        self.group: Optional[int] = None
        self.time_slots: List[List[Any]] = []
        self.slot_duras: Optional[ndarray] = None

    def add_symbols(self, symbols: List[Any]) -> None:
        self.symbols.extend(symbols)
        self.symbols.sort(key=lambda s: s.x_center)
        for sym in symbols:
            if isinstance(sym, Voice):
                self.voices.append(sym)
            elif isinstance(sym, Clef):
                self.clefs.append(sym)
                self.has_clef = True
            elif isinstance(sym, Accidental):
                self.accidentals.append(sym)
            elif isinstance(sym, Rest):
                self.rests.append(sym)
            else:
                raise ValueError(f"Unexpected symbol type: {type(sym)}")
        self.voices.sort(key=lambda s: s.x_center)
        self.clefs.sort(key=lambda s: s.x_center)
        self.accidentals.sort(key=lambda s: s.x_center)
        self.rests.sort(key=lambda s: s.x_center)

    def has_key(self) -> bool:
        total_tracks = get_total_track_nums()
        start = total_tracks if self.at_beginning else 0
        chunk = self.symbols[start:start + total_tracks]
        return bool(chunk) and all(isinstance(s, Accidental) for s in chunk)

    def get_key(self) -> Key:
        if not self.accidentals or not self.has_key():
            return Key(0)
        track_nums = get_total_track_nums()
        start = track_nums if self.at_beginning else 0
        end = min(start + track_nums * 7 + 4, len(self.symbols))
        candidates: List[Accidental] = []
        for sym in self.symbols[start:end]:
            if not isinstance(sym, Accidental):
                break
            candidates.append(sym)
        if not candidates:
            return Key(0)
        track_counts = [0] * track_nums
        for acc in candidates:
            track_counts[acc.track] += 1
        all_same = all(a.label == candidates[0].label for a in candidates)
        acc_label = candidates[0].label
        if not all_same:
            counter: Dict[AccidentalType, int] = {}
            for acc in candidates:
                counter[acc.label] = counter.get(acc.label, 0) + 1
            sorted_c = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            top = sorted_c[0][0]
            second = sorted_c[1][0] if len(sorted_c) > 1 else None
            if top == AccidentalType.NATURAL and second is not None:
                top = second
            acc_label = top
        count = round(sum(track_counts) / track_nums)
        if acc_label == AccidentalType.FLAT:
            count = -count
        for acc in candidates:
            acc.is_key = True
        return Key(count)


    def get_track_clef(self, instrument: InstrumentConfig) -> List[Optional[Clef]]:
        """Clef cho từng track — fallback dùng instrument.default_clefs."""
        if not (self.at_beginning or self.double_barline):
            return [None] * instrument.num_staves
        result: List[Optional[Clef]] = []
        for track in range(instrument.num_staves):
            matching = [c for c in self.clefs if c.track == track]
            if matching:
                result.append(matching[0])
            else:
                fallback = Clef()
                fallback.track = track
                fallback.group = self.group
                fallback.label = instrument.default_clefs[track]
                result.append(fallback)
        return result

    def align_symbols(self) -> None:
        track_nums = get_total_track_nums()
        unit_size = get_global_unit_size()
        time_slots: List[List[Any]] = []
        corr_sidx:  List[int] = []
        last_x: Optional[float] = None
        for idx, sym in enumerate(self.symbols):
            if isinstance(sym, (Clef, Accidental)):
                continue
            if last_x is None or abs(sym.x_center - last_x) >= unit_size:
                time_slots.append([sym])
                corr_sidx.append(idx)
                last_x = sym.x_center
            else:
                time_slots[-1].append(sym)
        track_duras = np.zeros((len(time_slots), track_nums), dtype=np.uint16)
        for si, slot in enumerate(time_slots):
            per_track: List[List[int]] = [[] for _ in range(track_nums)]
            for sym in slot:
                per_track[sym.track].append(_get_duration(sym))
            for t, durations in enumerate(per_track):
                track_duras[si, t] = min(durations) if durations else 0
        self.time_slots = time_slots
        self.slot_duras = track_duras
        if track_nums >= 2:
            self._balance_tracks(time_slots, corr_sidx, track_duras, track_nums)

    def _balance_tracks(
        self,
        time_slots: List[List[Any]],
        corr_sidx: List[int],
        track_duras: ndarray,
        track_nums: int,) -> None:
        """Cân bằng N tracks — không assert cứng == 2."""
        if track_nums == 2:
            self._balance_pair(time_slots, corr_sidx, track_duras, 0, 1)
        else:
            for t in range(track_nums - 1):
                self._balance_pair(time_slots, corr_sidx, track_duras, t, t + 1)

    def _balance_pair(
        self,
        time_slots: List[List[Any]],
        corr_sidx: List[int],
        track_duras: ndarray,
        track_a: int,
        track_b: int,) -> None:
        diff    = 0
        lead    = track_a
        add_idx = 0
        added   = 0
        solved  = True

        def apply_fix(slot_idx: int, deficit: int, follower: int, inserted: int) -> int:
            pos_dura = int(track_duras[slot_idx, follower])
            if pos_dura:
                for sym in time_slots[slot_idx]:
                    if sym.track == follower:
                        _extend_symbol_length(sym, deficit + pos_dura)
                track_duras[slot_idx, follower] = deficit + pos_dura
            else:
                rest = _make_rest(deficit)
                rest.track = follower
                rest.group = self.group
                cx = time_slots[slot_idx][0].x_center
                rest.bbox  = (cx, 0, cx, 0)
                ins_pos = corr_sidx[slot_idx] + inserted
                self.symbols.insert(ins_pos, rest)
                time_slots[slot_idx].insert(0, rest)
                track_duras[slot_idx, follower] = deficit
                inserted += 1
            return inserted

        for idx in range(len(track_duras)):
            ta = int(track_duras[idx, track_a])
            tb = int(track_duras[idx, track_b])
            if ta > 0 and tb > 0:
                if diff > 0:
                    follower = track_b if lead == track_a else track_a
                    added = apply_fix(add_idx, diff, follower, added)
                diff = abs(ta - tb)
                lead = track_a if ta >= tb else track_b
                add_idx = idx
                solved  = True
            elif ta > 0:
                if lead == track_a:
                    diff += ta
                elif diff >= ta:
                    diff -= ta
                else:
                    diff = ta - diff if diff else ta
                    add_idx = idx
                    lead = track_a
                solved = False
            elif tb > 0:
                if lead == track_b:
                    diff += tb
                elif diff >= tb:
                    diff -= tb
                else:
                    diff = tb - diff if diff else tb
                    add_idx = idx
                    lead = track_b
                solved = False

        if not solved and diff > 0:
            follower = track_b if lead == track_a else track_a
            apply_fix(add_idx, diff, follower, added)

        self.slot_duras = track_duras


    def get_time_slot_dura(self, x_center: float) -> Tuple[int, ndarray]:
        for idx, slot in enumerate(self.time_slots[:-1]):
            if slot[0].x_center <= x_center < self.time_slots[idx + 1][0].x_center:
                return idx, self.slot_duras[idx]
        return len(self.time_slots) - 1, self.slot_duras[-1]

    def __repr__(self) -> str:
        return f"Measure {self.number}  symbols={len(self.symbols)}"


# XML element builders
def build_note_element(
    note: NoteHead,
    clef_type: ClefType,
    ctx: ActionContext,
    is_chord: bool = False,
    voice_num: int  = 1,) -> Optional[Element]:
    if note.invalid:
        return None

    chroma = _get_chroma_pitch(note.staff_line_pos, clef_type)
    ctx_acc = ctx.accidental_state.get(chroma)
    if note.accidental is not None and note.accidental != ctx_acc:
        ctx.accidental_state[chroma] = note.accidental
    else:
        note.accidental = ctx_acc

    elem = Element('note')
    if is_chord:
        elem.append(Element('chord'))

    pitch = SubElement(elem, 'pitch')
    step = SubElement(pitch, 'step')
    alter = SubElement(pitch, 'alter')
    octave = SubElement(pitch, 'octave')
    alter.text = '0'

    table = _CLEF_PITCH_TABLE[clef_type]
    oct_offset = _CLEF_OCT_OFFSET[clef_type]
    pitch_offset = _CLEF_PITCH_OFFSET[clef_type]
    pos = int(note.staff_line_pos)
    step.text = table[pos % 7] if pos >= 0 else table[(-pos) % 7]

    if pos - pitch_offset >= 0:
        oct_val = math.floor((pos + pitch_offset) / 7) + oct_offset
    else:
        oct_val = -math.ceil((pos + pitch_offset) / -7) + oct_offset
    octave.text = str(oct_val)

    if note.accidental is not None:
        alter.text = ACCIDENTAL_TO_ALTER[note.accidental]

    if (oct_val < 0 or oct_val > 8
            or (oct_val == 0 and step.text != 'A')
            or (oct_val == 8 and step.text != 'C')):
        return None

    dur_val = NOTE_TYPE_TO_RHYTHM[note.label]['duration']
    if note.has_dot:
        dur_val = round(dur_val * 1.5)
    SubElement(elem, 'duration').text = str(dur_val)
    SubElement(elem, 'type').text     = NOTE_TYPE_TO_RHYTHM[note.label]['name']

    if note.has_dot:
        elem.append(Element('dot'))

    if note.accidental is not None:
        SubElement(elem, 'accidental').text = ACCIDENTAL_TO_NAME[note.accidental]

    SubElement(elem, 'stem').text  = 'up' if note.stem_up else 'down'
    SubElement(elem, 'staff').text = str(note.track + 1)
    SubElement(elem, 'voice').text = str(voice_num)
    return elem


def build_rest_element(rest: Rest) -> Element:
    elem = Element('note')
    is_whole_measure = rest.label in (RestType.WHOLE, RestType.WHOLE_HALF)
    if is_whole_measure:
        elem.append(Element('rest', attrib={'measure': 'yes'}))
    else:
        elem.append(Element('rest'))
    dur_val = REST_TYPE_TO_DURATION[rest.label]
    if rest.has_dot:
        dur_val = round(dur_val * 1.5)
    SubElement(elem, 'duration').text = str(dur_val)
    SubElement(elem, 'type').text = REST_TYPE_TO_NAME[rest.label]
    if rest.has_dot:
        elem.append(Element('dot'))
    SubElement(elem, 'staff').text = str(rest.track + 1)
    SubElement(elem, 'voice').text = '1'
    return elem


def build_backup_element(duration: int) -> Optional[Element]:
    if duration <= 0:
        return None
    e = Element('backup')
    SubElement(e, 'duration').text = str(duration)
    return e


def build_forward_element(duration: int) -> Optional[Element]:
    if duration <= 0:
        return None
    e = Element('forward')
    SubElement(e, 'duration').text = str(duration)
    return e


def build_key_element(key: Key) -> Element:
    attr = Element('attributes')
    SubElement(SubElement(attr, 'key'), 'fifths').text = str(key.value)
    return attr


def build_clef_element(clef: Clef) -> Element:
    attr = Element('attributes')
    cc   = SubElement(attr, 'clef', attrib={'number': str(clef.track + 1)})
    SubElement(cc, 'sign').text = _CLEF_SIGN[clef.label]
    SubElement(cc, 'line').text = _CLEF_LINE[clef.label]
    return attr


def build_part_list_element(instrument: InstrumentConfig) -> Element:
    parts = Element('part-list')
    part  = SubElement(parts, 'score-part', attrib={'id': instrument.part_id})
    SubElement(part, 'part-name').text = instrument.name
    si = SubElement(part, 'score-instrument', attrib={'id': instrument.instrument_id})
    SubElement(si, 'instrument-name').text = instrument.name
    SubElement(si, 'instrument-sound').text = instrument.sound
    midi = SubElement(part, 'midi-instrument', attrib={'id': instrument.instrument_id})
    SubElement(midi, 'midi-channel').text = str(instrument.midi_channel)
    SubElement(midi, 'midi-program').text = str(instrument.midi_program)
    SubElement(midi, 'volume').text = '80'
    SubElement(midi, 'pan').text = '0'
    return parts


def build_work_element(title: Optional[str] = None) -> Element:
    work = Element('work')
    SubElement(work, 'work-title').text = title or 'Untitled'
    return work


# Measure builders
def build_initial_measure_element(
    measure: Measure,
    ctx: ActionContext,
    instrument: InstrumentConfig,) -> Element:
    ctx.key = measure.get_key()
    ctx.clefs = measure.get_track_clef(instrument)
    ctx.reset_accidentals()

    elem = Element('measure', attrib={'number': str(measure.number)})
    attr = SubElement(elem, 'attributes')
    SubElement(attr, 'divisions').text = str(DIVISION_PER_QUARTER)
    SubElement(SubElement(attr, 'key'), 'fifths').text = str(ctx.key.value)
    SubElement(attr, 'staves').text = str(instrument.num_staves)

    for track, clef in enumerate(ctx.clefs):
        if clef is None:
            continue
        cc = SubElement(attr, 'clef', attrib={'number': str(track + 1)})
        SubElement(cc, 'sign').text = _CLEF_SIGN[clef.label]
        SubElement(cc, 'line').text = _CLEF_LINE[clef.label]

    return elem


def build_measure_element(
    measure: Measure,
    ctx: ActionContext,
    instrument: InstrumentConfig,
    cur_key: Key,
    cur_clefs: List[Optional[Clef]],
    add_system_break: bool = False,) -> Tuple[Element, Key, List[Optional[Clef]]]:
    ctx.reset_accidentals()
    elem = Element('measure', attrib={'number': str(measure.number)})
    if add_system_break:
        SubElement(elem, 'print', attrib={'new-system': 'yes'})

    notes = layers.get_layer('notes')
    total_tracks = instrument.num_staves
    last_tidx = 0
    last_dura = 0
    last_pos = 0
    track_duras = [0] * total_tracks
    min_dura_consumed = [False] * total_tracks

    for sym in measure.symbols:
        track = sym.track

        if isinstance(sym, Clef):
            prev = cur_clefs[track]
            if prev is None or prev.label != sym.label:
                elem.append(build_clef_element(sym))
                cur_clefs[track] = sym
                ctx.clefs[track] = sym
            continue

        if isinstance(sym, Accidental):
            if measure.has_key():
                new_key = measure.get_key()
                if new_key != cur_key:
                    elem.append(build_key_element(new_key))
                    cur_key = new_key
                    ctx.key = new_key
                    ctx.reset_accidentals()
            continue

        tidx, min_duras = measure.get_time_slot_dura(sym.x_center)
        min_dura = int(min_duras[track])
        dura = _get_duration(sym)

        if tidx == last_tidx and last_dura > 0:
            _append_if(elem, build_backup_element(int(last_dura)))
        else:
            min_dura_consumed = [False] * total_tracks
            diff = last_pos - track_duras[track]
            if diff > 0 and diff != last_dura:
                _append_if(elem, build_backup_element(int(diff)))

        if dura == min_dura and not min_dura_consumed[track]:
            track_duras[track] += min_dura
            min_dura_consumed[track] = True
            is_voice_one             = True
        else:
            is_voice_one = False

        if isinstance(sym, Rest):
            elem.append(build_rest_element(sym))
            last_pos = track_duras[track]
            last_dura = dura
            last_tidx = tidx
            continue

        if not isinstance(sym, Voice):
            continue

        voice_num = 1 if (sym.duration == min_dura and is_voice_one) else 2

        if voice_num == 1:
            _emit_voice(elem, sym, notes, ctx, cur_clefs, voice_num=1)
            last_pos  = track_duras[track]
            last_dura = dura
            last_tidx = tidx
        else:
            _emit_voice(elem, sym, notes, ctx, cur_clefs, voice_num=2)
            cur_pos = track_duras[track] + sym.duration
            if min_dura_consumed[track]:
                cur_pos -= min_dura
            diff = last_pos - cur_pos
            if diff > 0:
                _append_if(elem, build_forward_element(int(diff)))
            elif diff < 0:
                _append_if(elem, build_backup_element(int(-diff)))

    return elem, cur_key, cur_clefs


def _emit_voice(
    parent:    Element,
    voice:     Voice,
    notes:     Any,
    ctx:       ActionContext,
    cur_clefs: List[Optional[Clef]],
    voice_num: int,) -> None:
    for i, nid in enumerate(voice.note_ids):
        note  = notes[nid]
        clef_type = ctx.effective_clef(note.track)
        note_elem = build_note_element(note=note, clef_type=clef_type, ctx=ctx, is_chord=(i > 0), voice_num=voice_num)
        if note_elem is not None:
            parent.append(note_elem)


# Voice extraction & symbol sorting
def build_voices() -> List[Voice]:
    groups = layers.get_layer('note_groups')
    notes = layers.get_layer('notes')
    voices: List[Voice] = []

    def _add(grp: NoteGroup, nids: List[int], stem_up: bool) -> None:
        valid = [nid for nid in nids if not notes[nid].invalid]
        if not valid:
            return
        v = Voice()
        v.group = grp.group
        v.group_id = grp.id
        v.track = grp.track
        v.note_ids = valid
        v.stem_up = stem_up
        v.x_center = grp.x_center
        v.init()
        voices.append(v)

    for grp in groups:
        if grp.stem_up is None and grp.has_stem:
            _add(grp, grp.top_note_ids, True)
            _add(grp, grp.bottom_note_ids, False)
        else:
            _add(grp, grp.note_ids, grp.stem_up)

    for idx, v in enumerate(voices):
        v.id = idx
    return voices


def sort_symbols_by_group(voices: List[Voice]) -> Dict[int, List[Any]]:
    barlines = layers.get_layer('barlines')
    rests = layers.get_layer('rests')
    clefs = layers.get_layer('clefs')
    accidentals = layers.get_layer('accidentals')
    container: Dict[int, List[Any]] = {}

    def _add(items: List[Any]) -> None:
        for item in items:
            container.setdefault(item.group, []).append(item)

    _add(voices)
    _add(list(barlines))
    _add(list(rests))
    _add(list(clefs))
    _add([a for a in accidentals if a.note_id is None])

    for k in container:
        container[k].sort(key=lambda s: s.x_center)
    return dict(sorted(container.items()))


def build_measures_from_groups(
    group_container: Dict[int, List[Any]],
    instrument: InstrumentConfig,) -> Dict[int, List[Measure]]:
    measures: Dict[int, List[Measure]] = {}
    num = 1
    for grp, insts in group_container.items():
        measures[grp] = []
        buffer:        List[Any] = []
        at_beginning   = True
        double_barline = False
        for inst in insts:
            if isinstance(inst, Barline):
                if not buffer:
                    double_barline = True
                    continue
                mm = _make_measure(buffer, grp, num, at_beginning, double_barline, instrument)
                measures[grp].append(mm)
                num += 1
                buffer = []
                at_beginning = False
                double_barline = False
                continue
            buffer.append(inst)
        if buffer:
            mm = _make_measure(buffer, grp, num, at_beginning, double_barline, instrument)
            measures[grp].append(mm)
    return measures


def _make_measure(
    buffer: List[Any],
    group: int,
    number: int,
    at_beginning: bool,
    double_barline: bool,
    instrument: InstrumentConfig,) -> Measure:
    mm = Measure()
    mm.add_symbols(buffer)
    mm.double_barline = double_barline
    mm.number = number
    mm.at_beginning = at_beginning
    mm.group = group
    mm.get_key()
    mm.align_symbols()
    return mm


# XMLBuilder
class XMLBuilder:
    def __init__(
        self,
        title: Optional[str]  = None,
        instrument: Optional[InstrumentConfig] = None,) -> None:
        self.title = title
        self.instrument = instrument or INSTRUMENT_PRESETS['piano']
        self.measures: Dict[int, List[Measure]] = {}

    def build(self) -> None:
        voices = build_voices()
        group_container = sort_symbols_by_group(voices)
        self.measures = build_measures_from_groups(group_container, self.instrument)

    def to_xml(self, tempo: int = 90) -> bytes:
        if not self.measures:
            raise RuntimeError("Gọi build() trước khi gọi to_xml().")

        instrument = self.instrument
        ctx = ActionContext(instrument)

        score = Element('score-partwise', attrib={'version': '4.0'})
        score.append(build_work_element(self.title))
        score.append(build_part_list_element(instrument))
        part = SubElement(score, 'part', attrib={'id': instrument.part_id})
        SubElement(part, 'sound', attrib={'tempo': str(tempo)})

        cur_key: Key = Key(0)
        cur_clefs: List[Optional[Clef]] = [None] * instrument.num_staves
        first = True
        new_system = False

        for grp in sorted(self.measures.keys()):
            new_system = True
            for measure in self.measures[grp]:
                if first:
                    init_elem = build_initial_measure_element(measure, ctx, instrument)
                    cur_key   = ctx.key
                    cur_clefs = list(ctx.clefs)
                    part.append(init_elem)
                    body, cur_key, cur_clefs = build_measure_element(measure, ctx, instrument, cur_key, cur_clefs, add_system_break=False)
                    for child in list(body):
                        init_elem.append(child)
                    first = False
                else:
                    elem, cur_key, cur_clefs = build_measure_element(measure, ctx, instrument, cur_key, cur_clefs, add_system_break=new_system)
                    part.append(elem)
                new_system = False

        raw = _pretty_xml(score)
        doctype = (
            b'<!DOCTYPE score-partwise PUBLIC'
            b' "-//Recordare//DTD MusicXML 4.0 Partwise//EN"'
            b' "http://www.musicxml.org/dtds/partwise.dtd">'
        )
        parts = raw.split(b'?>')
        parts[0] += b'?>\n'
        parts.insert(1, doctype)
        result = b''.join(parts)
        result = re.sub(
            rb'\s+</measure>\n\s+<measure number="1">\n\s+<print[^/]*/>', b'', result
        )
        return result


# Internal helpers
def _get_duration(sym: Union[Voice, Rest]) -> int:
    if isinstance(sym, Voice):
        return sym.duration
    dura = REST_TYPE_TO_DURATION[sym.label]
    if sym.has_dot:
        dura = round(dura * 1.5)
    return dura


def _get_chroma_pitch(pos: int, clef_type: ClefType) -> str:
    table = _CLEF_PITCH_TABLE[clef_type]
    pos = int(pos)
    return table[pos % 7] if pos >= 0 else table[(-pos) % 7]


def _get_label_by_duration(
    duration: int,
    mapping:  Dict[Any, int],) -> Tuple[Any, bool]:
    best_label = None
    min_diff   = 10 ** 9
    for label, d in mapping.items():
        diff = duration - d
        if 0 <= diff < min_diff:
            min_diff   = diff
            best_label = label
    return best_label, min_diff > 0


def _make_rest(duration: int) -> Rest:
    rest = Rest()
    label, dot = _get_label_by_duration(duration, REST_TYPE_TO_DURATION)
    rest.label = label
    rest.has_dot = dot
    return rest


def _extend_symbol_length(symbol: Union[Voice, Rest], duration: int) -> None:
    notes = layers.get_layer('notes')
    if isinstance(symbol, Voice):
        mapping = {k: v['duration'] for k, v in NOTE_TYPE_TO_RHYTHM.items()}
        label, has_dot = _get_label_by_duration(duration, mapping)
        symbol.label = label
        symbol.has_dot = has_dot
        symbol.duration = duration
        symbol.rhythm_name = NOTE_TYPE_TO_RHYTHM[label]['name']
        for nid in symbol.note_ids:
            notes[nid].force_set_label(label)
            notes[nid].has_dot = has_dot
    elif isinstance(symbol, Rest):
        label, has_dot = _get_label_by_duration(duration, REST_TYPE_TO_DURATION)
        symbol.label = label
        symbol.has_dot = has_dot


def _append_if(parent: Element, child: Optional[Element]) -> None:
    if child is not None:
        parent.append(child)


def _pretty_xml(elem: Element) -> bytes:
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent='  ', encoding='UTF-8')


if __name__ == '__main__':
    preset_name = sys.argv[1] if len(sys.argv) > 1 else 'piano'
    instrument = INSTRUMENT_PRESETS.get(preset_name, INSTRUMENT_PRESETS['piano'])
    # print(f"Exporting as: {instrument.name}")
    logger.info(f"Exporting as: {instrument.name}")

    builder = XMLBuilder(title='Test Score', instrument=instrument)
    builder.build()
    xml = builder.to_xml(tempo=90)

    out = f'output_{preset_name}.musicxml'
    with open(out, 'wb') as f:
        f.write(xml)
    # print(f"Written {len(xml)} bytes to {out}")
    logger.info(f"Written {len(xml)} bytes to {out}")