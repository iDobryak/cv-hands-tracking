from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]
FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]
HAND_METRIC_COUNT = 8
FINGER_REMAP_MIN = 27.0
FINGER_REMAP_MAX = 100.0


def _to_xyz_array(landmarks: Iterable) -> np.ndarray:
    points = []
    for lm in landmarks:
        if hasattr(lm, "x"):
            points.append([lm.x, lm.y, lm.z])
        else:
            points.append([lm[0], lm[1], lm[2]])
    return np.asarray(points, dtype=np.float32)


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 180.0
    cosine = float(np.dot(ba, bc) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _clamp_0_100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _finger_curl_score(points: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    ang1 = _angle_degrees(points[mcp], points[pip], points[dip])
    ang2 = _angle_degrees(points[pip], points[dip], points[tip])
    curl = ((180.0 - ang1) + (180.0 - ang2)) / 360.0 * 100.0
    # User scale: 0 = fully bent (fist), 100 = fully straight.
    return _clamp_0_100(100.0 - curl)


def _remap_finger_score(value: float) -> float:
    # Empirical calibration: values <=27 should be treated as fully bent (0%).
    if value <= FINGER_REMAP_MIN:
        return 0.0
    if value >= FINGER_REMAP_MAX:
        return 100.0
    span = FINGER_REMAP_MAX - FINGER_REMAP_MIN
    return _clamp_0_100((value - FINGER_REMAP_MIN) / span * 100.0)


def _deg90_to_percent(value: float) -> float:
    return _clamp_0_100((value + 90.0) / 180.0 * 100.0)


def _hand_tilts(points: np.ndarray) -> tuple[float, float]:
    wrist = points[0]
    index_mcp = points[5]
    pinky_mcp = points[17]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return 0.0, 0.0
    n = normal / norm

    # Forward/back tilt and side tilt in degrees, clamped to practical range.
    forward_back_deg = float(np.degrees(np.arctan2(float(n[1]), abs(float(n[2])) + 1e-6)))
    side_deg = float(np.degrees(np.arctan2(float(n[0]), abs(float(n[2])) + 1e-6)))
    forward_back_deg = float(np.clip(forward_back_deg, -90.0, 90.0))
    side_deg = float(np.clip(side_deg, -90.0, 90.0))
    return forward_back_deg, side_deg


def hand_metrics_0_100(landmarks: Iterable) -> list[float]:
    """Return [rotation, thumb, index, middle, ring, pinky, tilt_fb, tilt_side] in range 0..100."""
    points = _to_xyz_array(landmarks)
    if len(points) < 21:
        return [0.0] * HAND_METRIC_COUNT

    wrist = points[0]
    index_mcp = points[5]
    pinky_mcp = points[17]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    rotation = _clamp_0_100((50.0 + np.tanh(float(normal[2]) * 8.0) * 50.0))

    thumb = _remap_finger_score(_finger_curl_score(points, 1, 2, 3, 4))
    index = _remap_finger_score(_finger_curl_score(points, 5, 6, 7, 8))
    middle = _remap_finger_score(_finger_curl_score(points, 9, 10, 11, 12))
    ring = _remap_finger_score(_finger_curl_score(points, 13, 14, 15, 16))
    pinky = _remap_finger_score(_finger_curl_score(points, 17, 18, 19, 20))
    tilt_fb_deg, tilt_side_deg = _hand_tilts(points)
    tilt_fb = _deg90_to_percent(tilt_fb_deg)
    tilt_side = _deg90_to_percent(tilt_side_deg)

    return [rotation, thumb, index, middle, ring, pinky, tilt_fb, tilt_side]


@dataclass
class HandMetricSmoother:
    alpha: float = 0.35
    decay: float = 0.92
    _state: dict[str, np.ndarray] = field(default_factory=dict)

    def update(self, label: str, values: list[float] | None) -> list[float]:
        key = label.lower()
        if values is None:
            prev = self._state.get(key)
            if prev is None:
                return [0.0] * HAND_METRIC_COUNT
            decayed = prev * self.decay
            self._state[key] = decayed
            return decayed.tolist()

        current = np.asarray(values, dtype=np.float32)
        prev = self._state.get(key)
        if prev is None:
            smoothed = current
        else:
            smoothed = self.alpha * current + (1.0 - self.alpha) * prev
        smoothed = np.clip(smoothed, 0.0, 100.0)
        self._state[key] = smoothed
        return smoothed.tolist()
