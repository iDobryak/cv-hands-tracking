from __future__ import annotations

import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.capture.camera import CameraSource
from src.vision.mediapipe_engine import MediaPipeVisionEngine


class VisionRuntime(QThread):
    frame_ready = Signal(QImage, object, object, object, float)
    error = Signal(str)

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._running = True
        self._camera_index = camera_index

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        camera = CameraSource(camera_index=self._camera_index)
        engine = MediaPipeVisionEngine()
        frame_times: list[float] = []
        try:
            camera.start()
            while self._running:
                frame = camera.read(timeout=0.25)
                if frame is None:
                    continue

                processed = engine.process(frame)

                now = time.perf_counter()
                frame_times.append(now)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 0.0
                if len(frame_times) >= 2:
                    dt = frame_times[-1] - frame_times[0]
                    if dt > 0:
                        fps = (len(frame_times) - 1) / dt
                self._draw_runtime_overlay(processed.frame_bgr, fps, processed.diagnostics["keypoints"])
                rgb = cv2.cvtColor(processed.frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg, processed.metrics, processed.head_metrics, processed.diagnostics, fps)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            camera.stop()
            engine.close()

    @staticmethod
    def _draw_runtime_overlay(frame_bgr, fps: float, keypoints_label: str) -> None:
        h, w = frame_bgr.shape[:2]
        panel_w = min(420, int(w * 0.45))
        x0 = w - panel_w - 16
        y0 = 16
        cv2.rectangle(frame_bgr, (x0, y0), (w - 16, y0 + 68), (15, 15, 15), -1)
        cv2.rectangle(frame_bgr, (x0, y0), (w - 16, y0 + 68), (70, 70, 70), 1)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 90), 2)
        cv2.putText(
            frame_bgr,
            f"Keypoints: {keypoints_label}",
            (x0 + 10, y0 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
