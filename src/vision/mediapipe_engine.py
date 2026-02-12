from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.metrics.face_metrics import FaceMetricSmoother, head_metrics
from src.metrics.hand_metrics import HandMetricSmoother, hand_metrics_0_100


@dataclass
class ProcessedFrame:
    frame_bgr: np.ndarray
    metrics: dict[str, list[float]]
    head_metrics: list[float]
    diagnostics: dict[str, str]


class MediaPipeVisionEngine:
    """Face + hands inference and overlay drawing."""

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smoother = HandMetricSmoother(alpha=0.35, decay=0.90)
        self._face_smoother = FaceMetricSmoother(alpha=0.30, decay=0.90)

    def close(self) -> None:
        self._hands.close()
        self._face.close()

    def process(self, frame_bgr: np.ndarray, source_mirrored: bool = False) -> ProcessedFrame:
        frame_bgr = frame_bgr.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        hand_result = self._hands.process(image_rgb)
        face_result = self._face.process(image_rgb)
        image_rgb.flags.writeable = True

        metrics: dict[str, list[float] | None] = {"left": None, "right": None}
        face_data = self._draw_face(frame_bgr, face_result)
        self._draw_hands(frame_bgr, hand_result, metrics, source_mirrored=source_mirrored)

        left_values = self._smoother.update("left", metrics["left"])
        right_values = self._smoother.update("right", metrics["right"])
        if source_mirrored and face_data["head_values"] is not None:
            # Mirror changes horizontal orientation, so yaw sign must be inverted.
            face_data["head_values"][0] = -face_data["head_values"][0]
        head_values = self._face_smoother.update(face_data["head_values"])

        diagnostics = {
            "face": "detected" if face_data["face_detected"] else "lost",
            "left_hand": "detected" if metrics["left"] is not None else "lost",
            "right_hand": "detected" if metrics["right"] is not None else "lost",
            "keypoints": "nose, left_eye, right_eye, mouth",
        }

        return ProcessedFrame(
            frame_bgr=frame_bgr,
            metrics={"left": left_values, "right": right_values},
            head_metrics=head_values,
            diagnostics=diagnostics,
        )

    def _draw_hands(
        self,
        frame_bgr: np.ndarray,
        hand_result: Any,
        metrics: dict[str, list[float] | None],
        source_mirrored: bool,
    ) -> None:
        if not hand_result.multi_hand_landmarks or not hand_result.multi_handedness:
            return

        h, w = frame_bgr.shape[:2]
        for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            raw_label = handedness.classification[0].label.lower()
            label = self._normalize_handedness(raw_label, source_mirrored)
            if label not in ("left", "right"):
                continue

            self._mp_draw.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
            wrist = hand_landmarks.landmark[0]
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
            metrics[label] = hand_metrics_0_100(hand_landmarks.landmark)

    @staticmethod
    def _normalize_handedness(label: str, source_mirrored: bool) -> str:
        if not source_mirrored:
            return label
        if label == "left":
            return "right"
        if label == "right":
            return "left"
        return label

    def _draw_face(self, frame_bgr: np.ndarray, face_result: Any) -> dict[str, Any]:
        if not face_result.multi_face_landmarks:
            return {"face_detected": False, "head_values": None}

        h, w = frame_bgr.shape[:2]
        face = face_result.multi_face_landmarks[0]
        lms = face.landmark

        key_ids = {
            "nose": 1,
            "left_eye": 33,
            "right_eye": 263,
            "mouth": 13,
        }
        for name, idx in key_ids.items():
            p = lms[idx]
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(frame_bgr, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        face = head_metrics(face)
        head_values = [
            face["yaw_deg"],
            face["pitch_deg"],
            face["eye_open_pct"],
            face["smile_pct"],
            face["mouth_open_pct"],
        ]
        return {"face_detected": True, "head_values": head_values}
