from dataclasses import dataclass

from src.metrics.face_metrics import FaceMetricSmoother, _gaze_axis_from_eye, head_metrics


@dataclass
class _Point:
    x: float
    y: float
    z: float = 0.0


class _Face:
    def __init__(self, points):
        self.landmark = points


def _make_face() -> _Face:
    pts = [_Point(0.5, 0.5) for _ in range(468)]
    pts[33] = _Point(0.4, 0.4)
    pts[263] = _Point(0.6, 0.4)
    pts[1] = _Point(0.5, 0.5)
    pts[13] = _Point(0.5, 0.60)
    pts[14] = _Point(0.5, 0.66)
    pts[61] = _Point(0.44, 0.62)
    pts[291] = _Point(0.56, 0.62)
    pts[159] = _Point(0.4, 0.39)
    pts[145] = _Point(0.4, 0.43)
    pts[386] = _Point(0.6, 0.39)
    pts[374] = _Point(0.6, 0.43)
    return _Face(pts)


def _make_face_with_eyes(left_xy: tuple[float, float], right_xy: tuple[float, float]) -> _Face:
    face = _make_face()
    face.landmark[33] = _Point(left_xy[0], left_xy[1])
    face.landmark[263] = _Point(right_xy[0], right_xy[1])
    return face


def _make_face_with_iris() -> _Face:
    face = _make_face()
    while len(face.landmark) <= 477:
        face.landmark.append(_Point(0.5, 0.5))

    face.landmark[33] = _Point(0.40, 0.40)
    face.landmark[133] = _Point(0.50, 0.40)
    face.landmark[362] = _Point(0.50, 0.40)
    face.landmark[263] = _Point(0.60, 0.40)
    face.landmark[159] = _Point(0.45, 0.39)
    face.landmark[145] = _Point(0.45, 0.43)
    face.landmark[386] = _Point(0.55, 0.39)
    face.landmark[374] = _Point(0.55, 0.43)

    for idx in [468, 469, 470, 471, 472]:
        face.landmark[idx] = _Point(0.49, 0.42)
    for idx in [473, 474, 475, 476, 477]:
        face.landmark[idx] = _Point(0.59, 0.42)

    return face


def test_head_metrics_ranges() -> None:
    result = head_metrics(_make_face())
    assert -180.0 <= result["yaw_deg"] <= 180.0
    assert -180.0 <= result["pitch_deg"] <= 180.0
    assert 0.0 <= result["eye_open_pct"] <= 100.0
    assert 0.0 <= result["smile_pct"] <= 100.0
    assert 0.0 <= result["mouth_open_pct"] <= 100.0


def test_tilt_angle_convention() -> None:
    upright = head_metrics(_make_face_with_eyes((0.4, 0.4), (0.6, 0.4)))["pitch_deg"]
    right_tilt = head_metrics(_make_face_with_eyes((0.5, 0.3), (0.5, 0.5)))["pitch_deg"]
    left_tilt = head_metrics(_make_face_with_eyes((0.5, 0.5), (0.5, 0.3)))["pitch_deg"]
    upside_down = head_metrics(_make_face_with_eyes((0.6, 0.4), (0.4, 0.4)))["pitch_deg"]

    assert abs(upright) < 1e-6
    assert 89.0 <= right_tilt <= 91.0
    assert -91.0 <= left_tilt <= -89.0
    assert 179.0 <= upside_down <= 180.0


def test_tilt_negative_180_is_normalized_to_positive_180() -> None:
    pitch = head_metrics(_make_face_with_eyes((0.6, 0.400000000001), (0.4, 0.4)))["pitch_deg"]
    assert pitch == 180.0


def test_face_smoother_decay() -> None:
    smoother = FaceMetricSmoother(alpha=0.5, decay=0.8)
    first = smoother.update([20.0, -10.0, 70.0, 60.0, 50.0])
    missing = smoother.update(None)
    assert missing[0] == first[0] * 0.8


def test_face_smoother_handles_missing_initial_state() -> None:
    smoother = FaceMetricSmoother()
    assert smoother.update(None) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_face_smoother_blends_and_clips_ranges() -> None:
    smoother = FaceMetricSmoother(alpha=0.5)
    smoother.update([0.0, 0.0, 0.0, 0.0, 0.0])
    second = smoother.update([400.0, -400.0, 200.0, -10.0, 150.0])

    assert second[0] == 180.0
    assert second[1] == -180.0
    assert second[2] == 100.0
    assert second[3] == 0.0
    assert second[4] == 75.0


def test_head_metrics_gaze_uses_iris_landmarks() -> None:
    metrics = head_metrics(_make_face_with_iris())
    assert metrics["gaze_x"] > 0.0
    assert metrics["gaze_y"] > 0.0


def test_gaze_axis_handles_zero_and_out_of_range_span() -> None:
    assert _gaze_axis_from_eye(1.0, 0.2, 0.2) == 0.0
    assert _gaze_axis_from_eye(-10.0, 0.0, 1.0) == -1.0
    assert _gaze_axis_from_eye(10.0, 0.0, 1.0) == 1.0
