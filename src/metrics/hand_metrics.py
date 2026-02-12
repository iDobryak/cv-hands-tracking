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
    return _clamp_0_100(curl)


def hand_metrics_0_100(landmarks: Iterable) -> list[float]:
    """Return [rotation, thumb, index, middle, ring, pinky] in range 0..100."""
    points = _to_xyz_array(landmarks)
    if len(points) < 21:
        return [0.0] * 6

    wrist = points[0]
    index_mcp = points[5]
    pinky_mcp = points[17]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    rotation = _clamp_0_100((50.0 + np.tanh(float(normal[2]) * 8.0) * 50.0))

    thumb = _finger_curl_score(points, 1, 2, 3, 4)
    index = _finger_curl_score(points, 5, 6, 7, 8)
    middle = _finger_curl_score(points, 9, 10, 11, 12)
    ring = _finger_curl_score(points, 13, 14, 15, 16)
    pinky = _finger_curl_score(points, 17, 18, 19, 20)

    return [rotation, thumb, index, middle, ring, pinky]


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
                return [0.0] * 6
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

