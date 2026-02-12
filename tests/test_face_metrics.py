from dataclasses import dataclass

from src.metrics.face_metrics import FaceMetricSmoother, head_metrics


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


def test_head_metrics_ranges() -> None:
    result = head_metrics(_make_face())
    assert -180.0 <= result["yaw_deg"] <= 180.0
    assert -180.0 <= result["pitch_deg"] <= 180.0
    assert 0.0 <= result["eye_open_pct"] <= 100.0
    assert 0.0 <= result["smile_pct"] <= 100.0
    assert 0.0 <= result["mouth_open_pct"] <= 100.0


def test_face_smoother_decay() -> None:
    smoother = FaceMetricSmoother(alpha=0.5, decay=0.8)
    first = smoother.update([20.0, -10.0, 70.0, 60.0, 50.0])
    missing = smoother.update(None)
    assert missing[0] == first[0] * 0.8

