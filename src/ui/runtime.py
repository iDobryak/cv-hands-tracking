from __future__ import annotations

import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.capture.camera import CameraSource
from src.vision.mediapipe_engine import MediaPipeVisionEngine


class VisionRuntime(QThread):
    frame_ready = Signal(QImage, object, float)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        camera = CameraSource()
        engine = MediaPipeVisionEngine()
        frame_times: list[float] = []
        try:
            camera.start()
            while self._running:
                frame = camera.read(timeout=0.25)
                if frame is None:
                    continue

                processed = engine.process(frame)
                rgb = cv2.cvtColor(processed.frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

                now = time.perf_counter()
                frame_times.append(now)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 0.0
                if len(frame_times) >= 2:
                    dt = frame_times[-1] - frame_times[0]
                    if dt > 0:
                        fps = (len(frame_times) - 1) / dt
                self.frame_ready.emit(qimg, processed.metrics, fps)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            camera.stop()
            engine.close()

