from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.metrics.hand_metrics import HandMetricSmoother, hand_metrics_0_100


@dataclass
class ProcessedFrame:
    frame_bgr: np.ndarray
    metrics: dict[str, list[float]]


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

    def close(self) -> None:
        self._hands.close()
        self._face.close()

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        frame_bgr = frame_bgr.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        hand_result = self._hands.process(image_rgb)
        face_result = self._face.process(image_rgb)
        image_rgb.flags.writeable = True

        metrics: dict[str, list[float] | None] = {"left": None, "right": None}
        self._draw_face(frame_bgr, face_result)
        self._draw_hands(frame_bgr, hand_result, metrics)

        left_values = self._smoother.update("left", metrics["left"])
        right_values = self._smoother.update("right", metrics["right"])
        cv2.putText(
            frame_bgr,
            "Face: nose/eyes/mouth + expression | Hands: Left/Right landmarks",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )

        return ProcessedFrame(frame_bgr=frame_bgr, metrics={"left": left_values, "right": right_values})

    def _draw_hands(self, frame_bgr: np.ndarray, hand_result: Any, metrics: dict[str, list[float] | None]) -> None:
        if not hand_result.multi_hand_landmarks or not hand_result.multi_handedness:
            return

        h, w = frame_bgr.shape[:2]
        for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            label = handedness.classification[0].label.lower()
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

    def _draw_face(self, frame_bgr: np.ndarray, face_result: Any) -> None:
        if not face_result.multi_face_landmarks:
            return

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

        mouth_open = self._norm_dist(lms[13], lms[14], lms[33], lms[263]) > 0.18
        smile = self._norm_dist(lms[61], lms[291], lms[33], lms[263]) > 0.50
        brow = (
            self._norm_dist(lms[105], lms[159], lms[33], lms[263]) > 0.15
            and self._norm_dist(lms[334], lms[386], lms[33], lms[263]) > 0.15
        )
        exp_text = f"smile:{int(smile)} mouth_open:{int(mouth_open)} brow_raise:{int(brow)}"
        cv2.putText(frame_bgr, exp_text, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

    @staticmethod
    def _norm_dist(a: Any, b: Any, n1: Any, n2: Any) -> float:
        d_ab = np.hypot(a.x - b.x, a.y - b.y)
        d_n = np.hypot(n1.x - n2.x, n1.y - n2.y) + 1e-6
        return float(d_ab / d_n)

