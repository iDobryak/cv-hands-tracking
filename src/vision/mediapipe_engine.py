from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

from src.metrics.face_metrics import FaceMetricSmoother, head_metrics
from src.metrics.hand_metrics import HandMetricSmoother, hand_metrics_0_100

_MODEL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "mediapipe"
_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass
class ProcessedFrame:
    frame_bgr: np.ndarray
    metrics: dict[str, list[float]]
    metrics_valid: dict[str, list[bool]]
    head_metrics: list[float]
    diagnostics: dict[str, str]


@dataclass
class _FaceLandmarksAdapter:
    landmark: list[Any]


class MediaPipeVisionEngine:
    """Face + hands inference and overlay drawing."""

    def __init__(self) -> None:
        hand_model_path = self._ensure_model("hand_landmarker.task", _HAND_MODEL_URL)
        face_model_path = self._ensure_model("face_landmarker.task", _FACE_MODEL_URL)

        self._hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(face_model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        self._smoother = HandMetricSmoother(alpha=0.35, decay=0.90)
        self._face_smoother = FaceMetricSmoother(alpha=0.30, decay=0.90)

    def close(self) -> None:
        self._hands.close()
        self._face.close()

    def process(self, frame_bgr: np.ndarray, source_mirrored: bool = False) -> ProcessedFrame:
        frame_bgr = frame_bgr.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        hand_result = self._hands.detect(mp_image)
        face_result = self._face.detect(mp_image)

        metrics: dict[str, list[float] | None] = {"left": None, "right": None}
        metrics_valid: dict[str, list[bool] | None] = {"left": None, "right": None}
        face_data = self._draw_face(frame_bgr, face_result, source_mirrored=source_mirrored)
        self._draw_hands(frame_bgr, hand_result, metrics, metrics_valid, source_mirrored=source_mirrored)

        left_values = self._smoother.update("left", metrics["left"])
        right_values = self._smoother.update("right", metrics["right"])
        head_values = self._face_smoother.update(face_data["head_values"])

        diagnostics = {
            "face": "detected" if face_data["face_detected"] else "lost",
            "left_hand": "detected" if metrics["left"] is not None else "lost",
            "right_hand": "detected" if metrics["right"] is not None else "lost",
            "keypoints": "nose, left_eye, right_eye, mouth",
            "gaze": face_data.get("gaze_label", "-"),
            "eyes_open": face_data.get("eyes_open_label", "-"),
        }

        return ProcessedFrame(
            frame_bgr=frame_bgr,
            metrics={"left": left_values, "right": right_values},
            metrics_valid={
                "left": metrics_valid["left"] or [False] * 8,
                "right": metrics_valid["right"] or [False] * 8,
            },
            head_metrics=head_values,
            diagnostics=diagnostics,
        )

    def _draw_hands(
        self,
        frame_bgr: np.ndarray,
        hand_result: Any,
        metrics: dict[str, list[float] | None],
        metrics_valid: dict[str, list[bool] | None],
        source_mirrored: bool,
    ) -> None:
        if not hand_result.hand_landmarks or not hand_result.handedness:
            return

        h, w = frame_bgr.shape[:2]
        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            raw_label = self._extract_handedness_label(handedness)
            label = self._normalize_handedness(raw_label, source_mirrored)
            if label not in ("left", "right"):
                continue

            self._draw_hand_landmarks(frame_bgr, hand_landmarks)
            wrist = hand_landmarks[0]
            x = int(wrist.x * w)
            y = int(wrist.y * h)
            cv2.putText(
                frame_bgr,
                label.upper(),
                (x + 8, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255) if label == "left" else (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            metrics[label] = hand_metrics_0_100(hand_landmarks)
            metrics_valid[label] = self._hand_metric_validity(hand_landmarks)

    @staticmethod
    def _normalize_handedness(label: str, source_mirrored: bool) -> str:
        if not source_mirrored:
            return label
        if label == "left":
            return "right"
        if label == "right":
            return "left"
        return label

    @staticmethod
    def _extract_handedness_label(handedness_entries: Any) -> str:
        if not handedness_entries:
            return ""
        candidate = handedness_entries[0]
        label = (
            getattr(candidate, "category_name", None)
            or getattr(candidate, "display_name", None)
            or getattr(candidate, "label", None)
            or ""
        )
        return str(label).lower()

    @staticmethod
    def _draw_hand_landmarks(frame_bgr: np.ndarray, hand_landmarks: list[Any]) -> None:
        h, w = frame_bgr.shape[:2]
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            p1 = hand_landmarks[conn.start]
            p2 = hand_landmarks[conn.end]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (30, 180, 240), 2, cv2.LINE_AA)
        for p in hand_landmarks:
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame_bgr, (x, y), 3, (40, 255, 90), -1, cv2.LINE_AA)

    @staticmethod
    def _point_valid(landmark: Any) -> bool:
        # Tasks API does not always expose visibility; coordinate bounds are a practical validity proxy.
        return -0.15 <= float(landmark.x) <= 1.15 and -0.15 <= float(landmark.y) <= 1.15

    @classmethod
    def _group_valid(cls, hand_landmarks: list[Any], indices: list[int]) -> bool:
        return all(cls._point_valid(hand_landmarks[i]) for i in indices)

    @classmethod
    def _hand_metric_validity(cls, hand_landmarks: list[Any]) -> list[bool]:
        # [rotation, thumb, index, middle, ring, pinky, tilt_fb, tilt_side]
        rot = cls._group_valid(hand_landmarks, [0, 5, 17])
        thumb = cls._group_valid(hand_landmarks, [1, 2, 3, 4])
        index = cls._group_valid(hand_landmarks, [5, 6, 7, 8])
        middle = cls._group_valid(hand_landmarks, [9, 10, 11, 12])
        ring = cls._group_valid(hand_landmarks, [13, 14, 15, 16])
        pinky = cls._group_valid(hand_landmarks, [17, 18, 19, 20])
        tilt_fb = cls._group_valid(hand_landmarks, [0, 5, 17])
        tilt_side = cls._group_valid(hand_landmarks, [0, 5, 17])
        return [rot, thumb, index, middle, ring, pinky, tilt_fb, tilt_side]

    def _draw_face(self, frame_bgr: np.ndarray, face_result: Any, source_mirrored: bool) -> dict[str, Any]:
        if not face_result.face_landmarks:
            return {"face_detected": False, "head_values": None, "gaze_label": "-", "eyes_open_label": "-"}

        h, w = frame_bgr.shape[:2]
        lms = face_result.face_landmarks[0]
        if len(lms) <= 386:
            return {"face_detected": False, "head_values": None, "gaze_label": "-", "eyes_open_label": "-"}

        for conn in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            p1 = lms[conn.start]
            p2 = lms[conn.end]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (70, 70, 70), 1, cv2.LINE_AA)

        if source_mirrored:
            key_ids = {
                "nose": 1,
                "left_eye": 33,
                "right_eye": 263,
                "mouth": 13,
            }
        else:
            key_ids = {
                "nose": 1,
                "left_eye": 263,
                "right_eye": 33,
                "mouth": 13,
            }
        for name, idx in key_ids.items():
            p = lms[idx]
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(frame_bgr, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        face_values = head_metrics(_FaceLandmarksAdapter(landmark=lms))
        head_values = [
            face_values["yaw_deg"],
            face_values["pitch_deg"],
            face_values["eye_open_pct"],
            face_values["smile_pct"],
            face_values["mouth_open_pct"],
        ]
        gaze_label = self._format_gaze_label(
            float(face_values.get("gaze_x", 0.0)),
            float(face_values.get("gaze_y", 0.0)),
            source_mirrored=source_mirrored,
        )
        left_eye_open = float(face_values.get("left_eye_open_pct", 0.0))
        right_eye_open = float(face_values.get("right_eye_open_pct", 0.0))
        if source_mirrored:
            left_eye_open, right_eye_open = right_eye_open, left_eye_open
        eyes_open_label = f"L:{left_eye_open:.0f}% R:{right_eye_open:.0f}%"
        return {
            "face_detected": True,
            "head_values": head_values,
            "gaze_label": gaze_label,
            "eyes_open_label": eyes_open_label,
        }

    @staticmethod
    def _format_gaze_label(gaze_x: float, gaze_y: float, source_mirrored: bool) -> str:
        # Gaze is computed on the same frame that is shown to the user.
        # Do not invert horizontally again for mirrored mode.
        gx = gaze_x
        gy = gaze_y
        threshold = 0.22
        horizontal = "center"
        vertical = "center"
        if gx < -threshold:
            horizontal = "left"
        elif gx > threshold:
            horizontal = "right"
        if gy < -threshold:
            vertical = "up"
        elif gy > threshold:
            vertical = "down"

        if horizontal == "center" and vertical == "center":
            return "center"
        if horizontal == "center":
            return vertical
        if vertical == "center":
            return horizontal
        return f"{horizontal}-{vertical}"

    @staticmethod
    def _ensure_model(filename: str, url: str) -> Path:
        _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _MODEL_CACHE_DIR / filename
        if path.exists():
            return path

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with urlopen(url, timeout=60) as response, tmp_path.open("wb") as out:
                out.write(response.read())
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download model '{filename}' from '{url}'. "
                f"Download it manually and place at '{path}'."
            ) from exc

        tmp_path.replace(path)
        return path
