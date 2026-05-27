from typing import Tuple
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from src.core import layers
from src.extraction.staffline_extraction import Staff

DEFAULT_UNIT_SIZE = 12.0


def _sanitize_unit_size(value: float, fallback: float = DEFAULT_UNIT_SIZE) -> float:
    if np.isfinite(value) and value > 0:
        return float(value)
    return float(fallback)

def count(data, intervals):
    """Count elements in different intervals."""
    data = np.asarray(data)
    if data.size == 0:
        return []

    intervals = np.sort(np.asarray(intervals))
    edges = np.concatenate(([data.min()], intervals, [np.nextafter(data.max(), np.inf)]))
    hist, _ = np.histogram(data, bins=edges)
    return hist.tolist()


def find_closest_staffs(x: int, y: int) -> Tuple[Staff, Staff]:
    staffs = layers.get_layer('staffs').reshape(-1)
    diffs = sorted(staffs, key=lambda st: st - [x, y])

    if len(diffs) == 1:
        return diffs[0], diffs[0]
    if len(diffs) == 2:
        return diffs[0], diffs[1]

    first = diffs[0]
    candidates = diffs[1:3]
    closer_to_lower = abs(first.y_lower - y) <= abs(first.y_upper - y)

    if closer_to_lower:
        for st in candidates:
            if st.y_center > first.y_center:
                return first, st
    else:
        for st in candidates:
            if st.y_center < first.y_center:
                return first, st

    return first, first


def get_unit_size(x: int, y: int) -> float:
    try:
        st1, st2 = find_closest_staffs(x, y)
    except Exception:
        return DEFAULT_UNIT_SIZE

    st1_unit = _sanitize_unit_size(float(st1.unit_size), fallback=DEFAULT_UNIT_SIZE)
    st2_unit = _sanitize_unit_size(float(st2.unit_size), fallback=st1_unit)

    if st1.y_center == st2.y_center:
        return st1_unit

    if st1.y_upper <= y <= st1.y_lower:
        return st1_unit

    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    denom = dist1 + dist2
    if denom <= 0:
        return st1_unit
    weighted = (dist2 * st1_unit + dist1 * st2_unit) / denom
    return _sanitize_unit_size(float(weighted), fallback=st1_unit)


def get_global_unit_size() -> float:
    staffs = layers.get_layer('staffs').reshape(-1)
    sizes = [
        float(st.unit_size)
        for st in staffs
        if np.isfinite(float(st.unit_size)) and float(st.unit_size) > 0
    ]
    if not sizes:
        return DEFAULT_UNIT_SIZE
    return float(np.median(sizes))


def get_total_track_nums() -> int:
    staffs = layers.get_layer('staffs').reshape(-1)
    return len({st.track for st in staffs})


def remove_stems(data):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.morphologyEx(data.astype(np.uint8), cv2.MORPH_OPEN, ker)


def estimate_degree(points, **kwargs):
    """Accepts list of (x, y) coordinates."""
    points = np.asarray(points)
    model = LinearRegression(**kwargs)
    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    return float(np.degrees(np.arctan(model.coef_[0])))


def slope_to_degree(y_diff: int, x_diff: int) -> float:
    return float(np.degrees(np.arctan2(y_diff, x_diff)))