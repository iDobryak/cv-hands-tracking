from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np

from src.capture.base import VideoSource


class CameraSource(VideoSource):
    """Default webcam capture with a producer queue for low-latency reads."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.camera_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            raise RuntimeError("Failed to open default camera")

        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        while self._running and self._capture is not None:
            ok, frame = self._capture.read()
            if not ok:
                continue
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self._capture:
            self._capture.release()
            self._capture = None

