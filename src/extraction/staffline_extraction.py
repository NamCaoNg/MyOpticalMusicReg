import enum
import pickle
from typing import List, Any, Tuple, Union
from typing_extensions import Self

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from src.core import layers
from src.utils.logger import get_logger
from src.utils.draw_bbox import BBox, find_lines, get_bbox, get_center

logger = get_logger(__name__)

def _cached_stat(obj, cache_name: str, values, fn) -> float:
    val = getattr(obj, cache_name)
    if val is None:
        val = float(fn(values)) if values else 0.0
        setattr(obj, cache_name, val)
    return val

class LineLabel(enum.Enum):
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4


class Line:
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []
        self.label: Union[LineLabel, None] = None
        self._reset_cache()

    def _reset_cache(self) -> None:
        self._y_center: Union[float, None] = None
        self._y_upper: Union[float, None] = None
        self._y_lower: Union[float, None] = None
        self._x_center: Union[float, None] = None
        self._x_left: Union[float, None] = None
        self._x_right: Union[float, None] = None
        self._slope: Union[float, None] = None

    def _point_stat(self, cache_name: str, axis: int, fn) -> float:
        return _cached_stat(self, cache_name, [p[axis] for p in self.points], fn)

    def add_point(self, y: int, x: int) -> None:
        self.points.append((y, x))
        self._reset_cache()

    @property
    def y_center(self) -> float:
        return self._point_stat("_y_center", 0, np.mean)

    @y_center.setter
    def y_center(self, val):
        self._y_center = val

    @property
    def y_upper(self) -> Union[float, None]:
        return self._point_stat("_y_upper", 0, np.min)

    @y_upper.setter
    def y_upper(self, val):
        self._y_upper = val

    @property
    def y_lower(self) -> float:
        return self._point_stat("_y_lower", 0, np.max)

    @y_lower.setter
    def y_lower(self, val):
        self._y_lower = val

    @property
    def x_center(self) -> float:
        return self._point_stat("_x_center", 1, np.mean)

    @x_center.setter
    def x_center(self, val):
        self._x_center = val

    @property
    def x_left(self) -> float:
        return self._point_stat("_x_left", 1, np.min)

    @x_left.setter
    def x_left(self, val):
        self._x_left = val

    @property
    def x_right(self) -> float:
        return self._point_stat("_x_right", 1, np.max)

    @x_right.setter
    def x_right(self, val):
        self._x_right = val

    @property
    def slope(self) -> float:
        if self._slope is not None:
            return self._slope

        pts = np.array([[p[1], p[0]] for p in self.points], dtype=np.float64)
        if len(pts) < 2:
            self._slope = 0.0
            return self._slope

        model = PCA(n_components=2)
        model.fit(pts)

        vx, vy = model.components_[0]
        self._slope = float("inf") if abs(vx) < 1e-8 else float(vy / vx)
        return self._slope

    def __lt__(self, line: Self) -> bool:
        return self.y_center < line.y_center

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return "Line(\n" \
            f"\tPoint count: {len(self.points)}\n" \
            f"\tCenter: {self.y_center}\n" \
            f"\tUpper bound: {self.y_upper}\n" \
            f"\tLower bound: {self.y_lower}\n" \
            f"\tLabel: {self.label}\n" \
            f"\tSlope: {self.slope}\n" \
            ")\n"


class Staff:
    def __init__(self) -> None:
        self.lines: List[Line] = []
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self.is_interp: bool = False
        self._reset_cache()

    def _reset_cache(self) -> None:
        self._y_center: Union[float, None] = None
        self._y_upper: Union[float, None] = None
        self._y_lower: Union[float, None] = None
        self._x_center: Union[float, None] = None
        self._x_left: Union[float, None] = None
        self._x_right: Union[float, None] = None
        self._unit_size: Union[float, None] = None
        self._slope: Union[float, None] = None

    def _line_stat(self, cache_name: str, attr: str, fn) -> float:
        return _cached_stat(self, cache_name, [getattr(line, attr) for line in self.lines], fn)

    def add_line(self, line: Line) -> None:
        self.lines.append(line)
        self._reset_cache()

    @property
    def y_center(self) -> float:
        return self._line_stat("_y_center", "y_center", np.mean)

    @property
    def y_upper(self) -> float:
        return self._line_stat("_y_upper", "y_upper", np.min)

    @property
    def y_lower(self) -> float:
        return self._line_stat("_y_lower", "y_lower", np.max)

    @property
    def x_center(self) -> float:
        return self._line_stat("_x_center", "x_center", np.mean)

    @property
    def x_left(self) -> float:
        return self._line_stat("_x_left", "x_left", np.min)

    @property
    def x_right(self) -> float:
        return self._line_stat("_x_right", "x_right", np.max)

    @property
    def unit_size(self) -> float:
        if self._unit_size is None:
            centers = np.array([line.y_center for line in self.lines], dtype=np.float64)
            gaps = np.diff(centers)
            self._unit_size = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
        return self._unit_size

    @property
    def incomplete(self) -> bool:
        return len(self.lines) != 5

    @property
    def slope(self) -> float:
        if self._slope is None:
            self._slope = float(np.mean([l.slope for l in self.lines])) if self.lines else 0.0
        return self._slope

    def duplicate(self, x_offset=0, y_offset=0):
        st = Staff()
        for line in self.lines:
            new_l = Line()
            new_l.label = line.label
            new_l.points = [(y + y_offset, x + x_offset) for y, x in line.points]
            new_l._reset_cache()
            st.lines.append(new_l)
        st._reset_cache()
        return st

    def __lt__(self, st):
        return self.y_center < st.y_center

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        return "Staff(\n" \
            f"\tLines: {len(self.lines)}\n" \
            f"\tCenter: {self.y_center}\n" \
            f"\tUpper bound: {self.y_upper}\n" \
            f"\tLower bound: {self.y_lower}\n" \
            f"\tUnit size: {self.unit_size}\n" \
            f"\tTrack: {self.track}\n" \
            f"\tGroup: {self.group}\n" \
            f"\tIs interpolation: {self.is_interp}\n" \
            f"\tSlope: {self.slope}\n" \
            ")\n"

    def __sub__(self, st: Union["Staff", Tuple[int, int], List[int]]) -> float:
        if isinstance(st, Staff):
            x, y = st.x_center, st.y_center
        else:
            x, y = st
        x_dist = (x - self.x_center) ** 2
        y_dist = (y - self.y_center) ** 2
        return float((x_dist + y_dist) ** 0.5)


def init_zones(staff_pred: ndarray, splits: int) -> Tuple[ndarray, int, int, int]:
    ys, xs = np.where(staff_pred > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([], dtype=object), 0, staff_pred.shape[1], staff_pred.shape[0]

    accum_x = np.sum(staff_pred, axis=0).astype(np.float64)
    mean_acc = np.mean(accum_x)
    if mean_acc > 1e-8:
        accum_x = accum_x / mean_acc

    half = round(len(accum_x) / 2)
    right_bound = min(int(np.max(xs)) + 50, staff_pred.shape[1])
    left_bound = max(int(np.min(xs)) - 50, 0)

    for i in range(half + 10, len(accum_x)):
        if np.mean(accum_x[i - 10:i]) < 0.1:
            right_bound = i
            break

    for i in range(half - 10, 0, -1):
        if np.mean(accum_x[i:i + 10]) < 0.1:
            left_bound = i
            break

    bottom_bound = min(int(np.max(ys)) + 100, len(staff_pred))
    width = right_bound - left_bound
    if width <= 0:
        return np.array([], dtype=object), left_bound, right_bound, bottom_bound

    step_size = max(1, round(width / splits))
    zones = []
    for start in range(left_bound, right_bound, step_size):
        end = start + step_size
        if right_bound - end < step_size:
            end = right_bound
            zones.append(range(start, end))
            break
        zones.append(range(start, end))

    return np.array(zones, dtype=object), left_bound, right_bound, bottom_bound


def _is_consistent_staffs(
    staffs: List[Staff],
    horizontal_diff_th: float,
    unit_size_diff_th: float) -> bool:
    line_num = [len(staff.lines) for staff in staffs]
    if len(set(line_num)) != 1:
        # print(f"Some of the stafflines contains less or more than 5 lines: {line_num}")
        logger.warning(f"Some of the stafflines contains less or more than 5 lines: {line_num}")
        return False

    centers = np.array([staff.y_center for staff in staffs], dtype=np.float64)
    if np.mean(centers) == 0:
        return False
    center_norm = np.abs(centers / np.mean(centers) - 1)
    if not np.all(center_norm < horizontal_diff_th):
        print(f"Centers of staff parts at the same row not aligned (Th: {horizontal_diff_th}): {center_norm}")
        return False

    unit_size = np.array([staff.unit_size for staff in staffs], dtype=np.float64)
    if np.mean(unit_size) == 0:
        return False
    unit_norm = np.abs(unit_size / np.mean(unit_size) - 1)
    if not np.all(unit_norm < unit_size_diff_th):
        print(f"Unit sizes not consistent (th: {unit_size_diff_th}): {unit_norm}")
        return False

    return True


def staff_extract(
    splits: int = 8,
    line_threshold: float = 0.8,
    horizontal_diff_th: float = 0.1,
    unit_size_diff_th: float = 0.1,
    barline_min_degree: int = 75) -> Tuple[ndarray, ndarray]:

    staff_pred = layers.get_layer('staff_pred')
    zones, *_ = init_zones(staff_pred, splits=splits)

    if len(zones) == 0:
        return np.array([], dtype=object), zones

    all_staffs: List[List[Staff]] = []
    for rr in zones:
        print(rr[0], rr[-1], end=' ')
        rr = np.array(rr, dtype=np.int64)
        staffs = extract_part(staff_pred[:, rr], x_offset=int(rr[0]), line_threshold=line_threshold)
        if staffs is not None and len(staffs) > 0:
            all_staffs.append(staffs)
            print(len(staffs))

    if not all_staffs:
        return np.array([], dtype=object), zones

    aligned_staffs: np.ndarray = align_staffs(all_staffs)

    num_track = further_infer_track_nums(aligned_staffs, min_degree=barline_min_degree)
    logger.debug(f"Tracks: {num_track}")

    for col_sts in aligned_staffs:
        for idx, st in enumerate(col_sts):
            st.track = idx % num_track
            st.group = idx // num_track

    if not all(len(staff) == len(aligned_staffs[0]) for staff in aligned_staffs):
        raise Exception("Staffline count is inconsistent across zones")

    valid_staffs: List[List[Staff]] = []
    for staffs in aligned_staffs.T:
        if _is_consistent_staffs(list(staffs), horizontal_diff_th, unit_size_diff_th):
            valid_staffs.append(list(staffs))

    return np.array(valid_staffs, dtype=object).T, zones


def extract_part(pred: ndarray, x_offset: int, line_threshold: float = 0.8) -> Union[List[Staff], None]:
    lines, _ = extract_line(pred, x_offset=x_offset, line_threshold=line_threshold)

    if len(lines) < 5 or len(lines) % 5 != 0:
        return None

    for i, line in enumerate(lines):
        expected = LineLabel(i % 5)
        if line.label != expected:
            logger.warning(f"Unexpected line label order: got={line.label}, expected={expected}")
            return None

    staffs: List[Staff] = []
    for i in range(0, len(lines), 5):
        st = Staff()
        st.lines = list(lines[i:i + 5])
        st._reset_cache()
        staffs.append(st)

    return staffs


def extract_line(pred: ndarray, x_offset: int, line_threshold: float = 0.8) -> Tuple[ndarray, ndarray]:
    sub_ys, sub_xs = np.where(pred > 0)
    if len(sub_ys) == 0:
        return np.array([], dtype=object), np.zeros(pred.shape[0], dtype=np.float64)

    count = np.bincount(sub_ys, minlength=len(pred)).astype(np.uint16)

    count = np.insert(count, [0, len(count)], [0, 0])
    std = np.std(count)
    if std < 1e-8:
        return np.array([], dtype=object), np.zeros(len(pred), dtype=np.float64)

    norm = (count - np.mean(count)) / std
    centers, _ = find_peaks(norm, height=line_threshold, distance=8, prominence=1)

    if len(centers) == 0:
        return np.array([], dtype=object), norm[1:-1]

    centers -= 1
    norm = norm[1:-1]

    valid_centers, groups = filter_line_peaks(centers, norm)
    cc = centers[valid_centers]
    if len(cc) < 2:
        return np.array([], dtype=object), norm

    gaps = np.diff(cc)
    max_gap = float(np.mean(np.sort(gaps)[:min(3, len(gaps))]))
    if max_gap <= 0:
        return np.array([], dtype=object), norm

    lines = [Line() for _ in range(len(centers))]
    line_points = [[] for _ in range(len(centers))]

    for y, x in zip(sub_ys, sub_xs):
        closest_cen = int(np.argmin(np.abs(centers - y)))
        cen = centers[closest_cen]
        if valid_centers[closest_cen] \
                and (norm[y] > min(line_threshold, 1.2)) \
                and (abs(y - cen) < max_gap):
            line_points[closest_cen].append((int(y), int(x + x_offset)))

    for line, pts in zip(lines, line_points):
        line.points = pts
        line._reset_cache()

    last_group = groups[0] if groups else 0
    cur_line_id = 0
    pack = sorted(zip(lines, valid_centers, groups), key=lambda obj: obj[0])

    for line, valid, grp in pack:
        if not valid:
            continue
        if grp != last_group:
            cur_line_id = 0
            last_group = grp

        if cur_line_id < 5:
            line.label = LineLabel(cur_line_id)
        cur_line_id += 1

    lines = np.array(lines, dtype=object)[valid_centers]
    return lines, norm


def filter_line_peaks(peaks: ndarray, norm: ndarray, max_gap_ratio: float = 1.5) -> Tuple[ndarray, List[int]]:
    if len(peaks) == 0:
        return np.array([], dtype=bool), []

    valid_peaks = np.ones(len(peaks), dtype=bool)

    for idx, p in enumerate(peaks):
        if norm[p] > 15:
            valid_peaks[idx] = False

    if len(peaks) == 1:
        return valid_peaks, [0]

    gaps = np.diff(peaks)
    count = max(5, round(len(peaks) * 0.2))
    approx_unit = float(np.mean(np.sort(gaps)[:min(count, len(gaps))]))
    max_gap = approx_unit * max_gap_ratio

    ext_peaks = [peaks[0] - max_gap - 1] + list(peaks)
    groups = []
    group = -1
    for i in range(1, len(ext_peaks)):
        if ext_peaks[i] - ext_peaks[i - 1] > max_gap:
            group += 1
        groups.append(group)

    groups.append(groups[-1] + 1)
    cur_g = groups[0]
    count = 1

    for idx in range(1, len(groups)):
        group = groups[idx]
        if group == cur_g:
            count += 1
            continue

        if count < 5:
            valid_peaks[idx - count:idx] = False
        elif count > 5:
            cand_peaks = peaks[idx - count:idx]
            head_part = cand_peaks[:5]
            tail_part = cand_peaks[-5:]
            if sum(norm[head_part]) > sum(norm[tail_part]):
                valid_peaks[idx - count + 5:idx] = False
            else:
                valid_peaks[idx - count:idx - 5] = False

        cur_g = group
        count = 1

    return valid_peaks, groups[:-1]


def _interpolate_staff(ref1: Staff, ref2: Staff, idx1: int, idx2: int, target_idx: int) -> Staff:
    if idx1 == idx2:
        return ref1.duplicate()

    ratio = (target_idx - idx1) / (idx2 - idx1)
    x_offset = (ref2.x_center - ref1.x_center) * ratio
    y_offset = (ref2.y_center - ref1.y_center) * ratio
    return ref1.duplicate(x_offset=x_offset, y_offset=y_offset)


def align_staffs(staffs: List[List[Staff]], max_dist_ratio: int = 3) -> ndarray:
    len_types = set(len(st_part) for st_part in staffs)
    if len(len_types) == 1:
        return np.array(staffs, dtype=object)

    max_len = max(len_types)
    grid = np.zeros((len(staffs), max_len), dtype=object)
    for idx, st_part in enumerate(staffs):
        if len(st_part) == max_len:
            grid[idx] = np.array(st_part, dtype=object)

    def get_nearby_sts(j, row):
        dists = [(idx, abs(idx - j)) for idx in range(len(row))]
        dists = sorted(dists, key=lambda it: it[1])
        idxs = [it[0] for it in dists]

        nearby_sts = []
        for near_idx in idxs:
            if isinstance(row[near_idx], Staff):
                nearby_sts.append((near_idx, row[near_idx]))
            if len(nearby_sts) >= 2:
                break
        return nearby_sts

    def get_nearest_ori_st(ref_st, ori_st_col):
        max_dist = ref_st.unit_size * max_dist_ratio
        for st in ori_st_col:
            dist = abs(st.y_center - ref_st.y_center)
            if dist < max_dist:
                return st
        return None

    for i in range(max_len):
        row = grid[:, i]
        for j, obj in enumerate(row):
            if isinstance(obj, Staff):
                continue

            ori_st_part = staffs[j]
            sts = get_nearby_sts(j, row)
            if len(ori_st_part) >= max_len or len(sts) == 0:
                continue

            ori_st = get_nearest_ori_st(sts[0][1], ori_st_part)
            if ori_st is not None:
                grid[j, i] = ori_st
                continue

            if len(sts) == 1:
                ref_idx, ref_st = sts[0]
                width = ref_st.x_right - ref_st.x_left
                x_offset = width * (j - ref_idx)
                new_st = ref_st.duplicate(x_offset=x_offset)
            else:
                (idx1, ref1), (idx2, ref2) = sts
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                    ref1, ref2 = ref2, ref1

                new_st = _interpolate_staff(ref1, ref2, idx1, idx2, j)

            new_st.is_interp = True
            grid[j, i] = new_st

    return grid


def further_infer_track_nums(staffs: ndarray, min_degree: int = 75) -> int:
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    clefs = layers.get_layer('clefs_keys_pred')

    mix = symbols - stems - notehead - clefs
    mix[mix < 0] = 0

    lines = find_lines(mix)
    lines = filter_lines(lines, staffs, min_degree=min_degree)
    bmap = get_barline_map(symbols, lines) + stems
    bmap[bmap > 1] = 1

    ker = np.ones((5, 2), dtype=np.uint8)
    ext_bmap = cv2.erode(cv2.dilate(bmap.astype(np.uint8), ker), ker)
    bboxes = get_bbox(ext_bmap)

    h_ratios = []
    for box in bboxes:
        h = box[3] - box[1]
        unit_size = naive_get_unit_size(staffs, *get_center(box))
        if h > unit_size and unit_size > 0:
            h_ratios.append(h / unit_size)

    h_ratios = np.array(h_ratios, dtype=np.float64)

    num_track = 1
    factor = 10
    for i in range(1, 10):
        valid_h = len(h_ratios[h_ratios > factor * i])
        if valid_h * (i + 1) > staffs.shape[1]:
            num_track += 1
        else:
            break
    return num_track


def get_degree(line: BBox) -> float:
    return float(np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0])))


def filter_lines(lines: List[BBox], staffs: ndarray, min_degree: int = 75) -> List[BBox]:
    flat_staffs = staffs.reshape(-1)
    min_y = min(st.y_upper for st in flat_staffs)
    min_x = min(st.x_left for st in flat_staffs)
    max_y = max(st.y_lower for st in flat_staffs)
    max_x = max(st.x_right for st in flat_staffs)

    cands = []
    for line in lines:
        degree = get_degree(line)
        if degree < min_degree:
            continue

        if line[1] < min_y \
                or line[3] > max_y \
                or line[0] < min_x \
                or line[2] > max_x:
            continue

        cands.append(line)
    return cands


def get_barline_map(symbols: ndarray, bboxes: List[BBox]) -> ndarray:
    img = np.zeros_like(symbols)
    for x1, y1, x2, y2 in bboxes:
        if x2 == x1:
            x2 += 1
        img[y1:y2, x1:x2] += symbols[y1:y2, x1:x2]
    img[img > 1] = 1
    return img


def naive_get_unit_size(staffs: ndarray, x: int, y: int) -> float:
    flat_staffs = staffs.reshape(-1)
    nearest = min(flat_staffs, key=lambda st: (st.x_center - x) ** 2 + (st.y_center - y) ** 2)
    return float(nearest.unit_size)


if __name__ == "__main__":
    f_name = "Nho_oi01"

    pred = pickle.load(open(f"./data/my_data/{f_name}.pkl", "rb"))['staff']
    layers.register_layer("staff_pred", pred)
    rr = range(1130, 1400)

    lines, norm = extract_line(pred[..., rr], 0)

    data = pred[..., rr]
    ys, xs = np.where(data > 0)
    count = np.bincount(ys, minlength=len(data)).astype(np.float64) if len(ys) > 0 else np.zeros(len(data))
    count = np.array([0] + list(count))

    std = np.std(count)
    norm = (count - np.mean(count)) / std if std > 1e-8 else np.zeros_like(count)

    threshold = 0.8
    peaks, _ = find_peaks(norm, height=threshold, distance=8, prominence=1)
    valid_peaks, groups = filter_line_peaks(peaks, norm)
    peaks = peaks[valid_peaks]

    plt.plot(norm)
    plt.plot(peaks, [threshold] * len(peaks), 'ro')
    plt.show()

    # Vẽ ảnh ma trận đè dòng kẻ
    if len(lines) > 0:
        plt.figure(figsize=(15, 8))
        plt.imshow(data, cmap='gray', aspect='auto')

        # 5 màu tương ứng với 5 nhãn dòng kẻ
        label_colors = {
            0: 'red',     # FIRST
            1: 'green',   # SECOND
            2: 'blue',    # THIRD
            3: 'cyan',    # FOURTH
            4: 'magenta'  # FIFTH
        }

        # print(f"Tìm thấy {len(lines)} staff...")
        logger.info(f"Tìm thấy {len(lines)} staff...")

        added_labels = set()

        for line in lines:
            if hasattr(line, 'points') and len(line.points) > 0:
                ys = [p[0] for p in line.points]
                xs = [p[1] for p in line.points]

                # Lấy giá trị nhãn từ class LineLabel
                if line.label is not None:
                    label_val = line.label.value
                    label_name = line.label.name
                else:
                    label_val = 0
                    label_name = "UNKNOWN"

                c = label_colors.get(label_val, 'yellow')

                # Chỉ vẽ label vào bảng chú thích 1 lần
                if label_name not in added_labels:
                    plt.scatter(xs, ys, color=c, s=1.5, label=label_name)
                    added_labels.add(label_name)
                else:
                    plt.scatter(xs, ys, color=c, s=1.5)

        plt.title("Kết quả phân loại dòng kẻ nhạc (Staff Lines) theo LineLabel")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.1, 1))
        plt.tight_layout()
        plt.show()
    else:
        logger.warning("Không thấy dòng staff nào.")