from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from src.capture.discovery import list_available_cameras


class SourceSelectDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Select Video Source")
        self.setModal(True)
        self.resize(360, 140)

        layout = QVBoxLayout(self)
        label = QLabel("Выберите источник видео для запуска:")
        layout.addWidget(label)

        self._combo = QComboBox()
        cameras = list_available_cameras()
        if not cameras:
            cameras = [0]
        for idx in cameras:
            self._combo.addItem(f"Webcam #{idx}", userData=idx)
        layout.addWidget(self._combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def camera_index(self) -> int:
        value = self._combo.currentData()
        return int(value) if value is not None else 0

