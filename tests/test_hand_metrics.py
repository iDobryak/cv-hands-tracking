import math
from dataclasses import dataclass

import numpy as np

from src.metrics.hand_metrics import (
    HandMetricSmoother,
    _angle_degrees,
    _hand_tilts,
    _remap_finger_score,
    hand_metrics_0_100,
)


@dataclass
class _Point:
    x: float
    y: float
    z: float


def _open_hand_landmarks():
    points = [(0.0, 0.0, 0.0)] * 21
    points[0] = (0.0, 0.0, 0.0)
    points[1] = (-0.1, -0.05, 0.0)
    points[2] = (-0.2, -0.10, 0.0)
    points[3] = (-0.3, -0.15, 0.0)
    points[4] = (-0.4, -0.20, 0.0)
    points[5] = (0.1, -0.1, -0.08)
    points[6] = (0.1, -0.2, -0.08)
    points[7] = (0.1, -0.3, -0.08)
    points[8] = (0.1, -0.4, -0.08)
    points[9] = (0.0, -0.1, -0.08)
    points[10] = (0.0, -0.2, -0.08)
    points[11] = (0.0, -0.3, -0.08)
    points[12] = (0.0, -0.4, -0.08)
    points[13] = (-0.1, -0.1, -0.08)
    points[14] = (-0.1, -0.2, -0.08)
    points[15] = (-0.1, -0.3, -0.08)
    points[16] = (-0.1, -0.4, -0.08)
    points[17] = (-0.2, -0.1, -0.08)
    points[18] = (-0.2, -0.2, -0.08)
    points[19] = (-0.2, -0.3, -0.08)
    points[20] = (-0.2, -0.4, -0.08)
    return points


def _fist_landmarks():
    points = _open_hand_landmarks()
    for tip_idx, base_idx in [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]:
        bx, by, bz = points[base_idx]
        points[tip_idx] = (bx + 0.02, by + 0.03, bz)
    return points


def test_hand_metrics_are_bounded() -> None:
    metrics = hand_metrics_0_100(_open_hand_landmarks())
    assert len(metrics) == 8
    assert all(0.0 <= v <= 100.0 for v in metrics)


def test_fist_has_more_curl_than_open_hand() -> None:
    open_values = hand_metrics_0_100(_open_hand_landmarks())
    fist_values = hand_metrics_0_100(_fist_landmarks())
    assert sum(open_values[1:6]) > sum(fist_values[1:6])


def test_hand_metrics_short_landmarks_return_zeroes() -> None:
    assert hand_metrics_0_100([(0.0, 0.0, 0.0)] * 5) == [0.0] * 8


def test_hand_metrics_accepts_object_landmarks() -> None:
    obj_points = [_Point(x, y, z) for x, y, z in _open_hand_landmarks()]
    metrics = hand_metrics_0_100(obj_points)
    assert len(metrics) == 8
    assert all(0.0 <= v <= 100.0 for v in metrics)


def test_angle_degrees_handles_zero_length_vectors() -> None:
    p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert _angle_degrees(p, p, p) == 180.0


def test_remap_finger_score_handles_lower_bound() -> None:
    assert _remap_finger_score(10.0) == 0.0


def test_hand_tilts_returns_zero_for_degenerate_plane() -> None:
    points = np.zeros((21, 3), dtype=np.float32)
    tilt_fb, tilt_side = _hand_tilts(points)
    assert tilt_fb == 0.0
    assert tilt_side == 0.0


def test_smoother_decay_on_missing_detection() -> None:
    smoother = HandMetricSmoother(alpha=0.5, decay=0.9)
    first = smoother.update("left", [100.0, 50.0, 40.0, 30.0, 20.0, 10.0, 70.0, 60.0])
    missing = smoother.update("left", None)
    assert math.isclose(first[0] * 0.9, missing[0], rel_tol=1e-5)


def test_smoother_missing_without_state_returns_zeroes() -> None:
    smoother = HandMetricSmoother()
    assert smoother.update("left", None) == [0.0] * 8


def test_smoother_blends_updates_case_insensitively() -> None:
    smoother = HandMetricSmoother(alpha=0.25)
    smoother.update("Left", [100.0] * 8)
    blended = smoother.update("left", [0.0] * 8)
    assert all(math.isclose(v, 75.0, rel_tol=1e-5) for v in blended)
