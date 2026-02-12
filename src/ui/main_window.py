from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget

from src.ui.runtime import VisionRuntime


class HandMetricsWidget(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #111; border: 1px solid #333; border-radius: 8px; }")

        layout = QVBoxLayout(self)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("QLabel { color: #e5e5e5; font-size: 15px; font-weight: 600; }")
        layout.addWidget(title_lbl)

        self.graph = pg.PlotWidget()
        self.graph.setBackground("#111")
        self.graph.setYRange(0, 100)
        self.graph.setMouseEnabled(x=False, y=False)
        self.graph.showGrid(x=True, y=True, alpha=0.20)
        self.graph.getAxis("left").setTextPen("#cccccc")
        self.graph.getAxis("bottom").setTextPen("#cccccc")
        self.graph.setMenuEnabled(False)
        self.graph.setXRange(-0.6, 5.6)

        labels = ["Rot", "Thumb", "Index", "Middle", "Ring", "Pinky"]
        axis = self.graph.getAxis("bottom")
        axis.setTicks([list(enumerate(labels))])

        self._bar = pg.BarGraphItem(x=list(range(6)), height=[0] * 6, width=0.7, brush="#3aa6ff")
        self.graph.addItem(self._bar)
        layout.addWidget(self.graph)

    def update_metrics(self, values: list[float]) -> None:
        self._bar.setOpts(height=values)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Realtime Face + Hand Tracking")
        self.resize(1280, 920)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.video_label = QLabel("Camera stream")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(640)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("QLabel { background: #090909; color: #aaaaaa; border: 1px solid #2a2a2a; }")
        root_layout.addWidget(self.video_label, stretch=4)

        info_row = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("QLabel { color: #dddddd; font-size: 13px; }")
        info_row.addWidget(self.fps_label)
        info_row.addStretch()
        root_layout.addLayout(info_row)

        bars_row = QHBoxLayout()
        self.left_widget = HandMetricsWidget("Left Hand Metrics")
        self.right_widget = HandMetricsWidget("Right Hand Metrics")
        bars_row.addWidget(self.left_widget)
        bars_row.addWidget(self.right_widget)
        root_layout.addLayout(bars_row, stretch=2)

        self.runtime = VisionRuntime()
        self.runtime.frame_ready.connect(self._on_frame_ready)
        self.runtime.error.connect(self._on_error)
        self.runtime.start()

    def _on_error(self, error: str) -> None:
        self.video_label.setText(f"Runtime error: {error}")

    def _on_frame_ready(self, image, metrics, fps: float) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.left_widget.update_metrics(metrics.get("left", [0.0] * 6))
        self.right_widget.update_metrics(metrics.get("right", [0.0] * 6))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.runtime.stop()
        super().closeEvent(event)


def run_app() -> None:
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

