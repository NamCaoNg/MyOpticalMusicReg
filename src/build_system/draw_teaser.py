from PIL import Image
from typing import List, Optional, Tuple

import cv2
import math
import numpy as np

from src.core import layers
from numpy import ndarray

from src.utils.draw_bbox import BBox


def draw_bbox(
    out: ndarray,
    bboxes: List[BBox],
    color: Tuple[int, int, int],
    text: Optional[str] = None,
    labels: Optional[List[str]] = None,
    text_y_pos: float = 1) -> None:
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        y_pos = y1 + round((y2 - y1) * text_y_pos)
        if text is not None:
            cv2.putText(out, text, (x2 + 2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        elif labels is not None:
            cv2.putText(out, labels[idx], (x2 + 2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


from src.extraction.symbol_extraction import ClefType

def _get_effective_clef_type(track: int):
    clefs = layers.get_layer('clefs')
    for c in clefs:
        if c.track == track:
            return c.label
    # return ClefType(track % 2 + 1)
    return ClefType.GCLEF if track % 2 == 0 else ClefType.FCLEF


def _staff_pos_to_step_oct(pos: int, track: int) -> Tuple[str, int]:
    clef_type = _get_effective_clef_type(track)

    if clef_type == ClefType.GCLEF:
        order = ['D', 'E', 'F', 'G', 'A', 'B', 'C']
        oct_offset = 4
        pitch_offset = 1
    elif clef_type == ClefType.CCLEF:
        order = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        oct_offset = 3
        pitch_offset = 2
    else:
        order = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
        oct_offset = 2
        pitch_offset = 3

    step = order[pos % 7] if pos >= 0 else order[pos % -7]

    if pos - pitch_offset >= 0:
        octave = math.floor((pos + pitch_offset) / 7) + oct_offset
    else:
        octave = -math.ceil((pos + pitch_offset) / -7) + oct_offset

    return step, int(octave)


def _acc_to_text(accidental) -> str:
    # accidental is usually AccidentalType.SHARP / AccidentalType.FLAT / None
    if accidental is None:
        return ""
    name = getattr(accidental, "name", str(accidental))
    if "SHARP" in name:
        return "#"
    if "FLAT" in name:
        return "b"
    return ""


def teaser() -> Image.Image:
    ori_img = layers.get_layer('original_image')
    notes = layers.get_layer('notes')
    groups = layers.get_layer('note_groups')
    barlines = layers.get_layer('barlines')
    clefs = layers.get_layer('clefs')
    accidentals = layers.get_layer('accidentals')
    rests = layers.get_layer('rests')

    out = np.copy(ori_img).astype(np.uint8)

    draw_bbox(out, [gg.bbox for gg in groups], color=(255, 192, 92), text="group")
    draw_bbox(out, [n.bbox for n in notes if not n.invalid], color=(194, 81, 167), labels=[str(n.label)[0] for n in notes if not n.invalid])
    draw_bbox(out, [b.bbox for b in barlines], color=(63, 87, 181), text='barline', text_y_pos=0.5)
    draw_bbox(out, [s.bbox for s in accidentals if s.note_id is None], color=(90, 0, 168), labels=[str(s.label.name) for s in accidentals if s.note_id is None])
    draw_bbox(out, [c.bbox for c in clefs], color=(235, 64, 52), labels=[c.label.name for c in clefs])
    draw_bbox(out, [r.bbox for r in rests], color=(12, 145, 0), labels=[r.label.name for r in rests])

    for note in notes:
        if note.label is not None:
            x1, y1, x2, y2 = note.bbox
            cv2.putText(out, note.label.name[0], (x2 + 2, y2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)

    return Image.fromarray(out)


# ---- Pitch teaser ----
def pitch() -> Image.Image:
    ori_img = layers.get_layer('original_image')
    notes = layers.get_layer('notes')
    groups = layers.get_layer('note_groups')
    barlines = layers.get_layer('barlines')
    clefs = layers.get_layer('clefs')
    accidentals = layers.get_layer('accidentals')
    rests = layers.get_layer('rests')

    out = np.copy(ori_img).astype(np.uint8)

    draw_bbox(out, [n.bbox for n in notes if not n.invalid], color=(255, 0, 0))
    draw_bbox(out, [b.bbox for b in barlines], color=(63, 87, 181), text_y_pos=0.5)
    draw_bbox(out, [s.bbox for s in accidentals if s.note_id is None], color=(90, 0, 168), labels=[str(s.label.name) for s in accidentals if s.note_id is None])
    draw_bbox(out, [c.bbox for c in clefs], color=(235, 64, 52), labels=[c.label.name for c in clefs])
    draw_bbox(out, [r.bbox for r in rests], color=(194, 81, 167), labels=[r.label.name for r in rests])

    for note in notes:
        if getattr(note, "invalid", False):
            continue
        if getattr(note, "staff_line_pos", None) is None:
            continue

        x1, y1, x2, y2 = note.bbox

        pos = int(note.staff_line_pos)
        track = int(getattr(note, "track", 0))
        step, octave = _staff_pos_to_step_oct(pos, track)
        acc = _acc_to_text(getattr(note, "accidental", None))

        txt = f"{step}{acc}{octave}"  # ví dụ: C4, F#5, Bb3
        cv2.putText(out, txt, (x2 + 2, y2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)

    return Image.fromarray(out)