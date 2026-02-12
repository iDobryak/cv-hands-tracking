from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _clip(value: float, vmin: float, vmax: float) -> float:
    return float(np.clip(value, vmin, vmax))


def _dist(a: Any, b: Any) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _norm_dist(a: Any, b: Any, n1: Any, n2: Any) -> float:
    return _dist(a, b) / (_dist(n1, n2) + 1e-6)


def head_metrics(face_landmarks: Any) -> dict[str, float]:
    """Return head metrics: yaw/pitch in degrees and expression percentages."""
    lms = face_landmarks.landmark
    left_eye = lms[33]
    right_eye = lms[263]
    nose = lms[1]
    mouth_up = lms[13]
    mouth_down = lms[14]
    mouth_left = lms[61]
    mouth_right = lms[291]
    left_eye_up = lms[159]
    left_eye_down = lms[145]
    right_eye_up = lms[386]
    right_eye_down = lms[374]

    eye_base = _dist(left_eye, right_eye) + 1e-6

    eye_mid_x = (left_eye.x + right_eye.x) * 0.5
    yaw_norm = (nose.x - eye_mid_x) / (eye_base * 0.5 + 1e-6)
    yaw_deg = _clip(yaw_norm * 90.0, -180.0, 180.0)

    # Roll/tilt angle convention:
    # upright -> 0, right tilt -> positive, left tilt -> negative, upside-down -> +180.
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
    if abs(tilt_deg + 180.0) < 1e-4:
        tilt_deg = 180.0
    pitch_deg = _clip(tilt_deg, -180.0, 180.0)

    left_eye_open = _clip(_dist(left_eye_up, left_eye_down) / (eye_base * 0.18) * 100.0, 0.0, 100.0)
    right_eye_open = _clip(_dist(right_eye_up, right_eye_down) / (eye_base * 0.18) * 100.0, 0.0, 100.0)
    eye_open_pct = (left_eye_open + right_eye_open) * 0.5

    smile_score = _clip((_dist(mouth_left, mouth_right) / (eye_base * 0.95) - 0.2) / 0.8 * 100.0, 0.0, 100.0)
    mouth_open_pct = _clip(_norm_dist(mouth_up, mouth_down, left_eye, right_eye) / 0.35 * 100.0, 0.0, 100.0)

    return {
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "eye_open_pct": eye_open_pct,
        "smile_pct": smile_score,
        "mouth_open_pct": mouth_open_pct,
    }


@dataclass
class FaceMetricSmoother:
    alpha: float = 0.30
    decay: float = 0.90
    _state: np.ndarray | None = field(default=None)

    def update(self, values: list[float] | None) -> list[float]:
        if values is None:
            if self._state is None:
                return [0.0] * 5
            self._state = self._state * self.decay
            return self._state.tolist()

        arr = np.asarray(values, dtype=np.float32)
        if self._state is None:
            self._state = arr
        else:
            self._state = self.alpha * arr + (1.0 - self.alpha) * self._state

        self._state[0] = np.clip(self._state[0], -180.0, 180.0)
        self._state[1] = np.clip(self._state[1], -180.0, 180.0)
        self._state[2:] = np.clip(self._state[2:], 0.0, 100.0)
        return self._state.tolist()
