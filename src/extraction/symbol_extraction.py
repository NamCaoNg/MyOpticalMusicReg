from typing import List, Union, Any, Tuple
import enum

import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from src.core import layers
from src.core.inference import predict
from src.utils.utils import get_global_unit_size, slope_to_degree, get_unit_size, find_closest_staffs
from src.utils.logger import get_logger
from src.utils.general_filtering_rules import filter_out_of_range_bbox, filter_out_small_area
from src.utils.draw_bbox import (
    BBox,
    merge_nearby_bbox,
    rm_merge_overlap_bbox,
    find_lines,
    draw_lines,
    draw_bounding_boxes,
    get_bbox,
    get_center,
    to_rgb_img)

logger = get_logger(__name__)

class ClefType(enum.Enum):
    GCLEF = 1
    CCLEF = 2
    FCLEF = 3


class AccidentalType(enum.Enum):
    FLAT = 1
    SHARP = 2
    NATURAL = 3


class RestType(enum.Enum):
    WHOLE_HALF = 1
    QUARTER = 2
    EIGHTH = 3
    SIXTEENTH = 4
    THIRTY_SECOND = 5
    SIXTY_FOURTH = 6
    WHOLE = 7
    HALF = 8


class Clef:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self._label: ClefType = None  # type: ignore

    @property
    def label(self) -> ClefType:
        return self._label

    @label.setter
    def label(self, val: ClefType) -> None:
        assert isinstance(val, ClefType)
        self._label = val

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Clef: {self.label.name} / Track: {self.track} / Group: {self.group}"


class Accidental:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.note_id: Union[int, None] = None
        self.is_key: Union[bool, None] = None  # Whether is key or accidental
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self._label: AccidentalType = None  # type: ignore

    @property
    def label(self) -> AccidentalType:
        return self._label

    @label.setter
    def label(self, val: AccidentalType) -> None:
        assert isinstance(val, AccidentalType)
        self._label = val

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Accidental: {self.label.name} / Note ID: {self.note_id} / Is key: {self.is_key}" \
            f" / Track: {self.track} / Group: {self.group}"


class Rest:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.has_dot: Union[bool, None] = None
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self._label: RestType = None  # type: ignore

    @property
    def label(self) -> RestType:
        return self._label

    @label.setter
    def label(self, val: RestType) -> None:
        assert isinstance(val, RestType)
        self._label = val

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Rest: {self.label.name} / Has dot: {self.has_dot} / Track: {self.track}" \
            f" / Group: {self.group}"


class Barline:
    def __init__(self) -> None:
        self.bbox: BBox = None  # type: ignore
        self.group: Union[int, None] = None

    @property
    def x_center(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    def __repr__(self):
        return f"Barline / Group: {self.group}"


CLEF_NAME_MAP = {
    "gclef": ClefType.GCLEF,
    "cclef": ClefType.CCLEF,
    "fclef": ClefType.FCLEF,
}

ACCIDENTAL_NAME_MAP = {
    "sharp": AccidentalType.SHARP,
    "flat": AccidentalType.FLAT,
    "natural": AccidentalType.NATURAL,
}

REST_NAME_MAP = {
    "rest_whole_half": RestType.WHOLE_HALF,
# ==========================================
    "rest_half": RestType.HALF,
    "rest_whole": RestType.WHOLE,
# ========================================
    "rest_quarter": RestType.QUARTER,
    "rest_8th": RestType.EIGHTH,
    "rest_16th": RestType.SIXTEENTH,
    "rest_32nd": RestType.THIRTY_SECOND,
    "rest_64th": RestType.SIXTY_FOURTH,
}


def get_staff(box: BBox):
    st, _ = find_closest_staffs(*get_center(box))
    return st


def predict_regions(mask: ndarray, bboxes: List[BBox], model_name: str) -> List[str]:
    labels = []
    for x1, y1, x2, y2 in bboxes:
        region = np.copy(mask[y1:y2, x1:x2])
        labels.append(predict(region, model_name))
    return labels


def filter_barlines(lines: List[BBox], min_height_unit_ratio: float = 3.75) -> ndarray:
    lines = filter_out_of_range_bbox(lines)
    lines = rm_merge_overlap_bbox(lines, mode='merge', overlap_ratio=0)

    # First round check, with line mode.
    valid_lines = []
    for line in lines:
        x1, y1, x2, y2 = line

        # Check slope. Degree should be within 80~100.
        deg = slope_to_degree(y2 - y1, x2 - x1)
        if abs(deg) < 75:
            continue

        valid_lines.append(line)

    if len(valid_lines) == 0:
        return np.empty((0, 4), dtype=int)

    # Second round check, in bbox mode.
    valid_lines = np.array(valid_lines)  # type: ignore
    max_x = np.max(valid_lines[:, 2])
    max_y = np.max(valid_lines[:, 3])
    data = np.zeros((max_y + 10, max_x + 10, 3))
    data = draw_lines(valid_lines, data, width=1)  # type: ignore
    boxes = get_bbox(data[..., 1])

    valid_box = []
    for box in boxes:
        _, y1, _, y2 = box
        unit_size = get_unit_size(*get_center(box))

        # Check height
        if (y2 - y1) < unit_size * min_height_unit_ratio:
            continue

        valid_box.append(box)

    if len(valid_box) == 0:
        return np.empty((0, 4), dtype=int)

    # Check overall height. Filter out height below threshold after norm.
    valid_box = sorted(valid_box, key=lambda box: box[3] - box[1])
    heights = np.array([b[3] - b[1] for b in valid_box])
    top_5 = np.mean(heights[-min(5, len(heights)):])
    norm = heights / top_5
    idx = np.where(norm > 0.5)[0]
    valid_box = np.array(valid_box)[idx]  # type: ignore

    return valid_box  # type: ignore


def parse_barlines(
    group_map: ndarray,
    stems_rests: ndarray,
    symbols: ndarray,
    min_height_unit_ratio: float = 3.75) -> ndarray:
    # Remove notehead from prediction
    barline_cand = np.where(stems_rests-group_map>1, 1, 0)

    # Straight lines and rest symbols
    no_note = np.where(symbols - group_map > 1, 1, 0)

    # Label each regions by connected pixels.
    bar_label, bnum = scipy.ndimage.label(barline_cand)
    sym_label, _ = scipy.ndimage.label(no_note)

    # Check overlapping with symbols prediction
    sym_barline_map = np.zeros_like(no_note)
    for i in range(1, bnum+1):
        idx = (bar_label == i)
        region = sym_label[idx]
        labels = set(np.unique(region))
        if 0 in labels:
            labels.remove(0)
        for label in labels:
            sym_idx = (sym_label == label)
            sym_barline_map[sym_idx] += no_note[sym_idx]
    sym_barline_map[sym_barline_map>0] = 1

    lines = find_lines(sym_barline_map)
    line_box = filter_barlines(lines, min_height_unit_ratio)
    logger.debug("Detected barlines: %d", len(line_box))

    return line_box


def filter_clef_box(bboxes: List[BBox]) -> List[BBox]:
    valid_box = []
    for box in bboxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        cen_x, cen_y = get_center(box)
        unit_size = get_unit_size(cen_x, cen_y)

        # Check size
        if w < unit_size*1.5 or h < unit_size*1.5:
            continue

        # Check position
        staff, _ = find_closest_staffs(cen_x, cen_y)
        if cen_y < staff.y_upper or cen_y > staff.y_lower:
            continue

        valid_box.append(box)
    return valid_box


def parse_clefs_keys(
    clefs_keys: ndarray,
    unit_size: float,
    clef_size_ratio: float = 3.5,
    max_clef_tp_ratio: float = 0.45) -> Tuple[List[BBox], List[BBox], List[str], List[str]]:
    global_cs = to_rgb_img(clefs_keys)  # type: ignore

    ker = np.ones((np.int64(unit_size//2), 1), dtype=np.uint8)
    clefs_keys = cv2.erode(cv2.dilate(clefs_keys.astype(np.uint8), ker), ker)
    bboxes = get_bbox(clefs_keys)
    bboxes = filter_out_of_range_bbox(bboxes)
    bboxes = rm_merge_overlap_bbox(bboxes, mode='merge', overlap_ratio=0.3)
    bboxes = filter_out_small_area(bboxes, area_size_func=lambda usize: usize**2)
    bboxes = merge_nearby_bbox(bboxes, unit_size*1.2)

    key_box = []
    clef_box = []
    for box in bboxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        region: ndarray = clefs_keys[box[1]:box[3], box[0]:box[2]]
        usize = get_unit_size(*get_center(box))
        area_size_ratio = w * h / usize**2
        area_tp_ratio = region[region>0].size / (w * h)
        if area_size_ratio > clef_size_ratio \
                and area_tp_ratio < max_clef_tp_ratio:
            clef_box.append(box)
        elif w > usize/2 and h > usize/2:
            key_box.append(box)

    clef_box = filter_clef_box(clef_box)

    clef_label = predict_regions(clefs_keys, clef_box, "clef")
    key_label = predict_regions(clefs_keys, key_box, "accidental")

    return clef_box, key_box, clef_label, key_label


def parse_rests(line_box: ndarray, unit_size: float) -> Tuple[List[BBox], List[str]]:
    stems_rests = layers.get_layer('stems_rests_pred')
    group_map = layers.get_layer('group_map')

    g_map = np.where(group_map>-1, 1, 0)

    data = np.zeros_like(group_map)
    for x1, y1, x2, y2 in line_box:
        data[y1:y2, x1:x2] = 1

    rests = stems_rests - g_map - data
    rests[rests<0] = 0

    temp = np.copy(rests)

    bboxes = get_bbox(rests)
    bboxes = filter_out_of_range_bbox(bboxes)
    if len(bboxes) == 0:
        return [], []
    bboxes = merge_nearby_bbox(bboxes, unit_size*1.2)
    bboxes = rm_merge_overlap_bbox(bboxes)
    bboxes = filter_out_small_area(bboxes, area_size_func=lambda usize: usize**2 * 0.7)
    temp = draw_bounding_boxes(bboxes, temp)

    label = []
    valid_box = []
    for box in bboxes:
        # Check size is valid
        if box[3] - box[1] > unit_size * 3.5 \
                or box[2] - box[0] < unit_size * 0.5:
            continue

        region = rests[box[1]:box[3], box[0]:box[2]]
        pred = predict(region, "rests")
        if "8th" in pred:
            pred = predict(region, "rests_above8")
        valid_box.append(box)
        label.append(pred)

    return valid_box, label


def gen_barlines(bboxes: ndarray) -> List[Barline]:
    barlines = []
    for box in bboxes:
        st = get_staff(box)
        b = Barline()
        b.bbox = box
        b.group = st.group
        barlines.append(b)
    return barlines


def gen_clefs(bboxes: List[BBox], labels: List[str]) -> List[Clef]:
    clefs = []
    for box, label in zip(bboxes, labels):
        st = get_staff(box)
        cc = Clef()
        cc.bbox = box
        cc.label = CLEF_NAME_MAP[label]
        cc.track = st.track
        cc.group = st.group
        clefs.append(cc)
    return clefs


def get_nearby_note_id(box: BBox, note_id_map: ndarray) -> Union[int, None]:
    cen_x, cen_y = get_center(box)
    unit_size = int(round(get_unit_size(cen_x, cen_y)))

    x1 = box[2]
    x2 = min(box[2] + unit_size, note_id_map.shape[1])
    y1 = max(0, cen_y - unit_size // 3)
    y2 = min(note_id_map.shape[0], cen_y + unit_size // 3 + 1)

    region = note_id_map[y1:y2, x1:x2]
    ids = region[region != -1]
    return int(ids.flat[0]) if ids.size > 0 else None


def gen_accidentals(bboxes: List[BBox], labels: List[str]) -> List[Accidental]:
    note_id_map = layers.get_layer('note_id')
    notes = layers.get_layer('notes')

    accidentals = []
    for box, label in zip(bboxes, labels):
        st = get_staff(box)
        ss = Accidental()
        ss.bbox = box
        ss.label = ACCIDENTAL_NAME_MAP[label]
        ss.note_id = get_nearby_note_id(box, note_id_map)
        ss.track = st.track
        ss.group = st.group
        ss.is_key = True

        if ss.note_id is not None:
            note = notes[ss.note_id]
            if ss.track != note.track:
                logger.warning("Track of accidental and note mismatch: %s\n%s", ss, note)
                notes[ss.note_id].invalid = True
            elif ss.group != note.group:
                logger.warning("Group of accidental and note mismatch: %s\n%s", ss, note)
                notes[ss.note_id].invalid = True
            else:
                notes[ss.note_id].accidental = ss.label
                ss.is_key = False

        accidentals.append(ss)
    return accidentals


def gen_rests(bboxes: List[BBox], labels: List[str]) -> List[Rest]:
    symbols = layers.get_layer('symbols_pred')

    rests = []
    for box, label in zip(bboxes, labels):
        st = get_staff(box)
        rr = Rest()
        rr.bbox = box
        rr.label = REST_NAME_MAP[label]
        rr.track = st.track
        rr.group = st.group
        rr.has_dot = False

        unit_size = int(round(get_unit_size(*get_center(box))))
        dot_range = range(box[2]+1, min(box[2]+unit_size, symbols.shape[1] - 1))
        dot_region = symbols[box[1]:box[3], dot_range]
        if 0 < np.sum(dot_region) < unit_size**2 / 7:
            rr.has_dot = True

        rests.append(rr)
    return rests


def symbol_extract(min_barline_h_unit_ratio: float = 3.75) -> Tuple[List[Barline], List[Clef], List[Accidental], List[Rest]]:
    # Fetch paramters
    symbols = layers.get_layer('symbols_pred')
    stems_rests = layers.get_layer('stems_rests_pred')
    clefs_keys = layers.get_layer('clefs_keys_pred')
    group_map = layers.get_layer('group_map')

    line_box = parse_barlines(group_map, stems_rests, symbols, min_height_unit_ratio=min_barline_h_unit_ratio)
    barlines = gen_barlines(line_box)

    unit_size = get_global_unit_size()
    clef_box, key_box, clef_label, key_label = parse_clefs_keys(clefs_keys, unit_size)
    clefs = gen_clefs(clef_box, clef_label)
    accidentals = gen_accidentals(key_box, key_label)

    rest_box, rest_label = parse_rests(line_box, unit_size)
    rests = gen_rests(rest_box, rest_label)

    return barlines, clefs, accidentals, rests


def draw_symbols(symbols, ori_img, labels=None, color=(235, 64, 52)):
    bboxes = [sym.bbox for sym in symbols]
    if labels is None:
        labels = [sym.label.name for sym in symbols]
    out = np.copy(ori_img)
    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(label), (x2+2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def show(data):
    plt.imshow(data, aspect='auto')
    plt.show()


if __name__ == "__main__":
    symbols = layers.get_layer('symbols_pred')
    stems_rests = layers.get_layer('stems_rests_pred')
    ori_img = layers.get_layer('original_image')
    clefs_keys = layers.get_layer('clefs_keys_pred')
    group_map = layers.get_layer('group_map')

    img = np.ones(group_map.shape+(3,)) * 255
    idx = np.where(stems_rests>0)
    img[idx[0], idx[1]] = 0

    barlines, clefs, Accidentals, rests = symbol_extract()

    aa = draw_symbols(clefs, ori_img)
    bb = draw_symbols(rests, aa, color=(11, 163, 0))
    cc = draw_symbols(Accidentals, bb, color=(53, 0, 168))
    dd = draw_bounding_boxes([b.bbox for b in barlines], cc, color=(250, 0, 200))  # type: ignore