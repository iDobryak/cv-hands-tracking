from __future__ import annotations

import cv2


def list_available_cameras(max_index: int = 6) -> list[int]:
    available: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            available.append(idx)
    return available

