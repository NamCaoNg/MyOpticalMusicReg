import enum
import typing
from typing import List, Tuple, Any, Union

import cv2
import scipy.ndimage
import numpy as np
from numpy import ndarray

from src.core import layers
from src.utils.draw_bbox import BBox, get_bbox, get_center, merge_nearby_bbox, rm_merge_overlap_bbox, to_rgb_img
from src.utils.utils import get_unit_size, find_closest_staffs, get_global_unit_size
from src.utils.logger import get_logger
from src.extraction.staffline_extraction import Staff

logger = get_logger(__name__)

class NoteHeadConstant:
    NOTEHEAD_MORPH_WIDTH_FACTOR = 0.5  # 0.444444
    NOTEHEAD_MORPH_HEIGHT_FACTOR = 0.4  # 0.37037
    NOTEHEAD_SIZE_RATIO = 1.285714  # width/height

    STEM_WIDTH_UNIT_RATIO = 0.272727
    STEM_HEIGHT_UNIT_RATIO = 4

    CLEF_ZONE_WIDTH_UNIT_RATIO = 4.5406916
    CLEF_WIDTH_UNIT_RATIO = 3.2173913
    SMALL_CLEF_WIDTH_UNIT_RATIO = 2.4347826

    STAFFLINE_WIDTH_UNIT_RATIO = 0.261


nhc = NoteHeadConstant


class NoteType(enum.Enum):
    WHOLE = 0
    HALF = 1
    QUARTER = 2
    EIGHTH = 3
    SIXTEENTH = 4
    THIRTY_SECOND = 5
    SIXTY_FOURTH = 6
    TRIPLET = 7
    OTHERS = 8

    # An intermediate product while parsing.
    HALF_OR_WHOLE = 9


class NoteHead:
    def __init__(self) -> None:
        self.points: List[tuple] = []
        self.pitch: Union[int, None] = None
        self.has_dot: bool = False
        self.bbox: BBox = None  # type: ignore
        self.stem_up: Union[bool, None] = None
        self.stem_right: Union[bool, None] = None
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self.staff_line_pos: int = None  # type: ignore
        self.invalid: Union[bool, None] = False
        self.id: Union[int, None] = None
        self.note_group_id: Union[int, None] = None
        self.accidental: Any = None  # See symbols_extraction.py

        # Protected attributes
        self._label: Union[NoteType, None] = None

    @property
    def label(self) -> Union[NoteType, None]:
        if self.invalid:
            logger.warning(f"Note {self.id} is not a valid note.")
            return None
        return self._label

    @label.setter
    def label(self, label: NoteType):
        if self._label is not None:
            logger.debug(
                f"The label has been set to: {self._label}."
                " Use 'force_set_label' instead if you really want to modify the label."
            )
            return
        self._label = label

    def force_set_label(self, label: NoteType):
        logger.debug(f"Force set label from {self.label} to {label}")
        assert isinstance(label, NoteType)
        self._label = label

    def add_point(self, x: int, y: int) -> None:
        self.points.append((y, x))

    def __lt__(self, nt):
        return self.staff_line_pos < nt.staff_line_pos

    def __repr__(self):
        return f"Notehead {self.id}(\n" \
            f"\tPoints: {len(self.points)}\n" \
            f"\tBounding box: {self.bbox}\n" \
            f"\tStem up: {self.stem_up}\n" \
            f"\tTrack: {self.track}\n" \
            f"\tGroup: {self.group}\n" \
            f"\tPitch: {self.pitch}\n" \
            f"\tDot: {self.has_dot}\n" \
            f"\tLabel: {self.label}\n" \
            f"\tStaff line pos: {self.staff_line_pos}\n" \
            f"\tIs valid: {not self.invalid}\n" \
            f"\tNote group ID: {self.note_group_id}\n" \
            f"\tSharp/Flat/Natural: {self.accidental}\n" \
            f")\n"


def morph_notehead(pred: ndarray, unit_size: float) -> ndarray:
    if not np.isfinite(unit_size) or unit_size <= 0:
        unit_size = 1.0

    small_size = max(1, int(round(unit_size / 3)))
    small_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_size, small_size))
    pred = cv2.erode(cv2.dilate(pred.astype(np.uint8), small_ker), small_ker)

    size = (
        max(1, int(round(unit_size * nhc.NOTEHEAD_MORPH_WIDTH_FACTOR))),
        max(1, int(round(unit_size * nhc.NOTEHEAD_MORPH_HEIGHT_FACTOR)))
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    img = cv2.erode(pred.astype(np.uint8), kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size[0] + 1, size[1] + 1))
    return cv2.dilate(img, kernel)


def adjust_bbox(bbox, noteheads):
    x1, y1, x2, y2 = bbox
    region = noteheads[y1:y2, x1:x2]
    ys = np.where(region > 0)[0]
    if ys.size == 0:
        # Invalid note. Will be eliminated with zero height.
        return None
    return (x1, y1 + ys.min() - 1, x2, y1 + ys.max() + 1)


def check_bbox_size(bbox: BBox, noteheads: ndarray, unit_size: float) -> List[BBox]:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cen_x, _ = get_center(bbox)
    note_w = nhc.NOTEHEAD_SIZE_RATIO * unit_size
    note_h = unit_size

    new_bbox = []
    if abs(w - note_w) > abs(w - note_w * 2):
        # Contains at least two notes, one left and one right.
        left_box = (bbox[0], bbox[1], cen_x, bbox[3])
        right_box = (cen_x, bbox[1], bbox[2], bbox[3])

        left_box = adjust_bbox(left_box, noteheads)
        right_box = adjust_bbox(right_box, noteheads)

        if left_box is not None:
            new_bbox.extend(check_bbox_size(left_box, noteheads, unit_size))
        if right_box is not None:
            new_bbox.extend(check_bbox_size(right_box, noteheads, unit_size))

    if new_bbox:
        result = []
        for box in new_bbox:
            result.extend(check_bbox_size(box, noteheads, unit_size))
        return result

    num_notes = int(round(h / note_h))
    if num_notes > 0:
        sub_h = h // num_notes
        return [
            (bbox[0], round(bbox[1] + i * sub_h), bbox[2], round(bbox[1] + (i + 1) * sub_h))
            for i in range(num_notes)
        ]
    return []


def filter_notehead_bbox(
    bboxes: List[BBox],
    notehead: ndarray,
    min_h_ratio: float = 0.4,
    max_h_ratio: int = 5,
    min_w_ratio: float = 0.3,
    max_w_ratio: int = 3,
    min_area_ratio: float = 0.5) -> List[BBox]:

    zones = layers.get_layer('zones')

    min_x = zones[0][0]
    max_x = zones[-1][-1]

    valid_bboxes = []
    for bbox in bboxes:
        cen_x, cen_y = get_center(bbox)
        unit_size = get_unit_size(cen_x, cen_y)

        if (cen_x < min_x + nhc.CLEF_ZONE_WIDTH_UNIT_RATIO * unit_size) or (cen_x > max_x):
            continue

        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        if (h < unit_size * min_h_ratio) or (h > unit_size * max_h_ratio):
            continue
        if (w < unit_size * min_w_ratio * nhc.NOTEHEAD_SIZE_RATIO) \
                or (w > unit_size * max_w_ratio * nhc.NOTEHEAD_SIZE_RATIO):
            continue

        region = notehead[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        min_count = h * w * min_area_ratio
        count = region[region > 0].size
        if count < min_count:
            continue

        valid_bboxes.append(bbox)
    return valid_bboxes


def get_notehead_bbox(
    pred: ndarray,
    global_unit_size: float,
    min_h_ratio: float = 0.4,
    max_h_ratio: int = 5,
    min_w_ratio: float = 0.3,
    max_w_ratio: int = 3,
    min_area_ratio: float = 0.6
) -> List[BBox]:

    logger.debug("Morph noteheads")
    note = morph_notehead(pred, unit_size=global_unit_size)
    bboxes = get_bbox(note)
    bboxes = rm_merge_overlap_bbox(bboxes)

    result_bboxes: List[BBox] = []
    for box in bboxes:
        unit_size = get_unit_size(*get_center(box))
        checked_boxes = check_bbox_size(box, pred, unit_size)  # type: ignore
        result_bboxes.extend(checked_boxes)

    logger.debug("Detected noteheads: %d", len(result_bboxes))

    logger.debug("Filtering noteheads")
    bboxes = filter_notehead_bbox(
        result_bboxes,
        note,
        min_h_ratio=min_h_ratio,
        max_h_ratio=max_h_ratio,
        min_w_ratio=min_w_ratio,
        max_w_ratio=max_w_ratio,
        min_area_ratio=min_area_ratio
    )
    logger.debug("Detected noteheads after filtering: %d", len(bboxes))
    return bboxes


def fill_hole(region: ndarray) -> ndarray:
    return scipy.ndimage.binary_fill_holes(region > 0).astype(np.uint8)


def gen_notes(bboxes: List[ndarray], symbols: ndarray) -> List[NoteHead]:
    notes = []
    for bbox in bboxes:
        nn = NoteHead()
        nn.bbox = typing.cast(BBox, bbox)

        region = symbols[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ys, xs = np.where(region > 0)
        nn.points = list(zip(ys + bbox[1], xs + bbox[0]))

        def assign_group_track(st: Staff) -> None:
            nn.group = st.group
            nn.track = st.track

        cen_x, cen_y = get_center(bbox)
        st1, st2 = find_closest_staffs(cen_x, cen_y)
        if (st1.y_center == st2.y_center) or (st1.y_upper <= cen_y <= st1.y_lower):
            assign_group_track(st1)
            st_master = st1
        else:
            up_st, lo_st = (st1, st2) if st1.y_center < st2.y_center else (st2, st1)
            sts_cen = (up_st.y_center + lo_st.y_center) / 2
            if cen_y < sts_cen:
                assign_group_track(up_st)
                st_master = up_st
            else:
                assign_group_track(lo_st)
                st_master = lo_st

        # Determine staffline position.
        step = st_master.unit_size / 2
        line_centers = [line.y_center for line in reversed(st_master.lines)]

        pos_cen = [line_centers[0] + step]
        for i, cen in enumerate(line_centers[:-1]):
            pos_cen.append(cen)
            pos_cen.append((cen + line_centers[i + 1]) / 2)
        pos_cen.append(line_centers[-1])
        pos_cen.append(line_centers[-1] - step)

        pos_idx = np.argmin(np.abs(np.array(pos_cen) - cen_y))
        if 0 < pos_idx < len(pos_cen) - 1:
            nn.staff_line_pos = int(pos_idx)
        elif pos_idx == 0:
            diff = abs(pos_cen[0] - cen_y)
            pos = round(diff / step)
            nn.staff_line_pos = -pos
        else:
            diff = abs(pos_cen[-1] - cen_y)
            pos = round(diff / step) + len(pos_cen) - 1
            nn.staff_line_pos = pos

        notes.append(nn)
    return notes


def parse_stem_info(notes: List[NoteHead]) -> None:
    stems = layers.get_layer('stems_rests_pred')

    ker = np.ones((3, 2), dtype=np.uint8)
    enhanced_stems = cv2.dilate(stems.astype(np.uint8), ker)
    st_map, num_stems = scipy.ndimage.label(enhanced_stems)

    stem_center_x = {}
    for label in range(1, num_stems + 1):
        xs = np.where(st_map == label)[1]
        if xs.size > 0:
            stem_center_x[label] = float(xs.mean())

    for note in notes:
        x1, y1, x2, y2 = note.bbox  # type: ignore
        region = st_map[y1:y2, x1:x2]
        labels = set(np.unique(region)) - {0}
        if not labels:
            continue

        label = next(iter(labels))
        note_center_x = (x1 + x2) / 2
        note.stem_right = stem_center_x[label] > note_center_x


def notehead_extract(
    min_h_ratio: float = 0.4,
    max_h_ratio: int = 5,
    min_w_ratio: float = 0.3,
    max_w_ratio: int = 3,
    min_area_ratio: float = 0.5,
    max_whole_note_width_factor: float = 1.5,
    y_dist_factor: int = 5,
    hollow_filled_ratio_th: float = 1.3
) -> List[NoteHead]:

    pred = layers.get_layer('notehead_pred')
    symbols = layers.get_layer('symbols_pred')

    unit_size = get_global_unit_size()
    logger.info("Analyzing notehead bboxes")
    bboxes = get_notehead_bbox(
        pred,
        unit_size,
        min_h_ratio=min_h_ratio,
        max_h_ratio=max_h_ratio,
        min_w_ratio=min_w_ratio,
        max_w_ratio=max_w_ratio,
        min_area_ratio=min_area_ratio
    )

    nn_img = to_rgb_img(pred)

    ## -- Special cases for half/whole notes -- ##
    merged_box = merge_nearby_bbox(
        bboxes,
        distance=unit_size * max_whole_note_width_factor,
        y_factor=y_dist_factor
    )
    solid_box = []
    hollow_box = []
    for box in merged_box:
        box = np.array(box) - 1  # type: ignore
        region = symbols[box[1]:box[3], box[0]:box[2]]
        count = region[region > 0].size
        if count == 0:
            continue

        filled = fill_hole(region)
        f_count = filled[filled > 0].size
        ratio = f_count / count

        cv2.rectangle(nn_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(nn_img, f"{ratio:.2f}", (box[2] + 2, box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        if ratio > hollow_filled_ratio_th:
            hollow_box.append(box)
        else:
            solid_box.append(box)

    logger.info("Instanitiating notes")
    solid_notes = gen_notes(solid_box, symbols)  # type: ignore
    hollow_notes = gen_notes(hollow_box, symbols)  # type: ignore

    logger.debug("Setting temporary note type")
    for idx in range(len(hollow_notes)):
        hollow_notes[idx].label = NoteType.HALF_OR_WHOLE

    logger.debug("Parsing whether stem is on the right")
    notes = solid_notes + hollow_notes
    parse_stem_info(notes)

    return notes


def draw_notes(notes, ori_img):
    img = ori_img.copy()
    img = np.array(img)
    for note in notes:
        x1, y1, x2, y2 = note.bbox
        x_offset = 0
        y_offset = 0
        cv2.rectangle(img, (x1 + x_offset, y1 + y_offset), (x2 + x_offset, y2 + y_offset), (0, 255, 0), 2)
        if note.label:
            cv2.putText(img, note.label.name[0], (x2 + 2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    return img


if __name__ == "__main__":
    f_name = "Home0001"

    staff = layers.get_layer('staff_pred')
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    ori_img = layers.get_layer('original_image')

    aa = np.ones(staff.shape + (3,)) * 255
    idx = np.where(notehead + stems > 0)
    aa[idx[0], idx[1]] = 0

    notes = notehead_extract()
    rr = draw_notes(notes, aa)