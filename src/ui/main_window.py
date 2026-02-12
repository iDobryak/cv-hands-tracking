from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.runtime import VisionRuntime
from src.ui.source_dialog import SourceSelectDialog


class MetricsBarWidget(QFrame):
    def __init__(
        self,
        title: str,
        labels: list[str],
        y_min: float,
        y_max: float,
        brush: str = "#3aa6ff",
        value_formatter=None,
    ) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #111; border: 1px solid #333; border-radius: 8px; }")
        self._labels = labels
        self._value_formatter = value_formatter or (lambda idx, value: f"{value:.1f}")
        self._y_min = y_min
        self._y_max = y_max

        layout = QVBoxLayout(self)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("QLabel { color: #e5e5e5; font-size: 15px; font-weight: 600; }")
        layout.addWidget(title_lbl)

        self.graph = pg.PlotWidget()
        self.graph.setBackground("#111")
        self.graph.setYRange(y_min, y_max)
        self.graph.setMouseEnabled(x=False, y=False)
        self.graph.showGrid(x=True, y=True, alpha=0.20)
        self.graph.getAxis("left").setTextPen("#cccccc")
        self.graph.getAxis("bottom").setTextPen("#cccccc")
        self.graph.setMenuEnabled(False)
        self.graph.setXRange(-0.6, len(labels) - 0.4)

        axis = self.graph.getAxis("bottom")
        axis.setTicks([list(enumerate(labels))])

        self._bar = pg.BarGraphItem(x=list(range(len(labels))), height=[0] * len(labels), width=0.7, brush=brush)
        self.graph.addItem(self._bar)
        self._value_items = []
        for idx in range(len(labels)):
            txt = pg.TextItem(text="0", color="#ffffff", anchor=(0.5, 1.1))
            txt.setPos(idx, 0.0)
            self.graph.addItem(txt)
            self._value_items.append(txt)
        layout.addWidget(self.graph)

    def update_metrics(self, values: list[float]) -> None:
        if len(values) != len(self._labels):
            values = values[: len(self._labels)] + [0.0] * max(0, len(self._labels) - len(values))
        self._bar.setOpts(height=values)
        for idx, value in enumerate(values):
            txt = self._value_formatter(idx, value)
            self._value_items[idx].setText(txt)
            offset = (self._y_max - self._y_min) * 0.03
            y_pos = value + offset if value >= 0 else value - offset
            self._value_items[idx].setPos(idx, y_pos)


class MainWindow(QMainWindow):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self.setWindowTitle("Realtime Face + Hand Tracking")
        self.resize(1460, 940)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        top_row = QHBoxLayout()
        root_layout.addLayout(top_row, stretch=4)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMaximumWidth(360)
        left_panel.setStyleSheet("QFrame { background: #101010; border: 1px solid #2c2c2c; border-radius: 8px; }")
        left_layout = QVBoxLayout(left_panel)
        self.source_label = QLabel(f"Источник: Webcam #{camera_index}")
        self.source_label.setStyleSheet("QLabel { color: #dedede; font-size: 14px; font-weight: 600; }")
        left_layout.addWidget(self.source_label)

        self.face_status = QLabel("Face: -")
        self.left_status = QLabel("Left hand: -")
        self.right_status = QLabel("Right hand: -")
        self.fps_label = QLabel("FPS: 0.0")
        for lbl in (self.face_status, self.left_status, self.right_status, self.fps_label):
            lbl.setStyleSheet("QLabel { color: #bdbdbd; font-size: 13px; }")
            left_layout.addWidget(lbl)

        self.head_widget = MetricsBarWidget(
            title="Head Metrics",
            labels=["Yaw", "Pitch", "Eye", "Smile", "Mouth"],
            y_min=-180.0,
            y_max=180.0,
            brush="#f28f3b",
            value_formatter=format_head_value,
        )
        left_layout.addWidget(self.head_widget, stretch=1)
        top_row.addWidget(left_panel, stretch=0)

        self.video_label = QLabel("Camera stream")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(640)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("QLabel { background: #090909; color: #aaaaaa; border: 1px solid #2a2a2a; }")
        top_row.addWidget(self.video_label, stretch=1)

        bars_row = QHBoxLayout()
        self.left_widget = MetricsBarWidget(
            title="Left Hand Metrics",
            labels=["Rot", "Thumb", "Index", "Middle", "Ring", "Pinky"],
            y_min=0.0,
            y_max=100.0,
            brush="#3aa6ff",
            value_formatter=lambda idx, v: f"{v:.0f}%",
        )
        self.right_widget = MetricsBarWidget(
            title="Right Hand Metrics",
            labels=["Rot", "Thumb", "Index", "Middle", "Ring", "Pinky"],
            y_min=0.0,
            y_max=100.0,
            brush="#66d17a",
            value_formatter=lambda idx, v: f"{v:.0f}%",
        )
        bars_row.addWidget(self.left_widget)
        bars_row.addWidget(self.right_widget)
        root_layout.addLayout(bars_row, stretch=2)

        self.runtime = VisionRuntime(camera_index=camera_index)
        self.runtime.frame_ready.connect(self._on_frame_ready)
        self.runtime.error.connect(self._on_error)
        self.runtime.start()

    def _on_error(self, error: str) -> None:
        self.video_label.setText(f"Runtime error: {error}")

    def _on_frame_ready(self, image, metrics, head_metrics, diagnostics, fps: float) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.face_status.setText(f"Face: {diagnostics.get('face', '-')}")
        self.left_status.setText(f"Left hand: {diagnostics.get('left_hand', '-')}")
        self.right_status.setText(f"Right hand: {diagnostics.get('right_hand', '-')}")
        self.head_widget.update_metrics(head_metrics)
        self.left_widget.update_metrics(metrics.get("left", [0.0] * 6))
        self.right_widget.update_metrics(metrics.get("right", [0.0] * 6))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.runtime.stop()
        super().closeEvent(event)


def run_app() -> None:
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    source_dialog = SourceSelectDialog()
    if source_dialog.exec() != source_dialog.DialogCode.Accepted:
        return
    win = MainWindow(camera_index=source_dialog.camera_index)
    win.show()
    sys.exit(app.exec())


def _head_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def format_head_value(index: int, value: float) -> str:
    if index in (0, 1):
        return f"{value:.0f}°"
    return f"{_head_percent(value):.0f}%"
