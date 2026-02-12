from __future__ import annotations

import signal
import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.layout_settings import LayoutSettings, load_layout_settings, save_layout_settings
from src.ui.runtime import VisionRuntime
from src.ui.source_dialog import SourceSelectDialog

_LAYOUT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / ".cache" / "layout_settings.json"


class MetricsBarWidget(QFrame):
    def __init__(
        self,
        title: str,
        labels: list[str],
        y_min: float,
        y_max: float,
        brush: str = "#3aa6ff",
        value_formatter=None,
        widget_height: int = 280,
    ) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #111; border: 1px solid #333; border-radius: 8px; }")
        self.setMinimumHeight(min(220, widget_height))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._labels = labels
        self._value_formatter = value_formatter or (lambda idx, value: f"{value:.1f}")
        self._y_min = y_min
        self._y_max = y_max
        self._active_brush = brush
        self._inactive_brush = "#6f6f6f"
        self._display_values = [0.0] * len(labels)

        layout = QVBoxLayout(self)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("QLabel { color: #e5e5e5; font-size: 15px; font-weight: 600; }")
        layout.addWidget(title_lbl)

        self.graph = pg.PlotWidget()
        self.graph.setBackground("#111")
        y_pad = (y_max - y_min) * 0.20
        self.graph.setYRange(y_min - y_pad * 0.45, y_max + y_pad)
        self.graph.setMouseEnabled(x=False, y=False)
        self.graph.showGrid(x=True, y=True, alpha=0.20)
        self.graph.getAxis("left").setTextPen("#cccccc")
        self.graph.getAxis("bottom").setTextPen("#cccccc")
        self.graph.setMenuEnabled(False)
        self.graph.setXRange(-0.6, len(labels) - 0.4)

        axis = self.graph.getAxis("bottom")
        axis.setTicks([list(enumerate(labels))])

        self._bar = pg.BarGraphItem(
            x=list(range(len(labels))),
            height=[0] * len(labels),
            width=0.7,
            brushes=[self._active_brush] * len(labels),
        )
        self.graph.addItem(self._bar)
        self._value_items = []
        for idx in range(len(labels)):
            txt = pg.TextItem(text="0", color="#ffffff", anchor=(0.5, 1.1))
            txt.setPos(idx, 0.0)
            self.graph.addItem(txt)
            self._value_items.append(txt)
        layout.addWidget(self.graph)

    def update_metrics(self, values: list[float], valid_mask: list[bool] | None = None) -> None:
        if len(values) != len(self._labels):
            values = values[: len(self._labels)] + [0.0] * max(0, len(self._labels) - len(values))
        if valid_mask is None or len(valid_mask) != len(self._labels):
            valid_mask = [True] * len(self._labels)

        brushes: list[str] = []
        for idx, (value, is_valid) in enumerate(zip(values, valid_mask)):
            if is_valid:
                self._display_values[idx] = value
                brushes.append(self._active_brush)
            else:
                brushes.append(self._inactive_brush)

        self._bar.setOpts(height=self._display_values, brushes=brushes)
        for idx, value in enumerate(self._display_values):
            txt = self._value_formatter(idx, value)
            self._value_items[idx].setText(txt)
            offset = (self._y_max - self._y_min) * 0.03
            y_pos = value + offset if value >= 0 else value - offset
            self._value_items[idx].setPos(idx, y_pos)


class MainWindow(QMainWindow):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self.setWindowTitle("Realtime Face + Hand Tracking")
        self._settings_path = _LAYOUT_SETTINGS_PATH
        self._layout_settings = load_layout_settings(self._settings_path)
        self.resize(self._layout_settings.window_width, self._layout_settings.window_height)
        self.setMinimumSize(960, 680)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setChildrenCollapsible(False)
        self.vertical_splitter.setHandleWidth(8)
        root_layout.addWidget(self.vertical_splitter, stretch=1)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.setHandleWidth(8)
        top_layout.addWidget(self.top_splitter)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(240)
        left_panel.setStyleSheet("QFrame { background: #101010; border: 1px solid #2c2c2c; border-radius: 8px; }")
        left_layout = QVBoxLayout(left_panel)
        self.source_label = QLabel(f"Источник: Webcam #{camera_index}")
        self.source_label.setStyleSheet("QLabel { color: #dedede; font-size: 14px; font-weight: 600; }")
        left_layout.addWidget(self.source_label)
        self.source_button = QPushButton("Сменить источник")
        self.source_button.setStyleSheet(
            "QPushButton { background: #1f1f1f; color: #d8d8d8; border: 1px solid #3a3a3a; padding: 6px 10px; }"
            "QPushButton:hover { background: #2a2a2a; }"
        )
        left_layout.addWidget(self.source_button)
        self.invert_checkbox = QCheckBox("Инвертировать камеру (зеркально)")
        self.invert_checkbox.setChecked(self._layout_settings.invert_camera)
        self.invert_checkbox.setStyleSheet("QCheckBox { color: #d2d2d2; font-size: 13px; }")
        left_layout.addWidget(self.invert_checkbox)

        self.face_status = QLabel("Face: -")
        self.gaze_status = QLabel("Gaze: -")
        self.left_status = QLabel("Left hand: -")
        self.right_status = QLabel("Right hand: -")
        self.fps_label = QLabel("FPS: 0.0")
        for lbl in (self.face_status, self.gaze_status, self.left_status, self.right_status, self.fps_label):
            lbl.setStyleSheet("QLabel { color: #bdbdbd; font-size: 13px; }")
            left_layout.addWidget(lbl)

        self.head_widget = MetricsBarWidget(
            title="Head Metrics",
            labels=["Yaw", "Tilt", "Eye", "Smile", "Mouth"],
            y_min=-180.0,
            y_max=180.0,
            brush="#f28f3b",
            value_formatter=format_head_value,
            widget_height=300,
        )
        left_layout.addStretch(1)
        left_layout.addWidget(self.head_widget, stretch=0, alignment=Qt.AlignmentFlag.AlignBottom)
        self.top_splitter.addWidget(left_panel)

        self.video_label = AspectRatioVideoLabel()
        self.video_label.setMinimumSize(320, 200)
        self.top_splitter.addWidget(self.video_label)
        self.top_splitter.setStretchFactor(0, 0)
        self.top_splitter.setStretchFactor(1, 1)

        hands_container = QWidget()
        hands_layout = QVBoxLayout(hands_container)
        hands_layout.setContentsMargins(0, 0, 0, 0)
        self.hands_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.hands_splitter.setChildrenCollapsible(False)
        self.hands_splitter.setHandleWidth(8)
        self.left_widget = MetricsBarWidget(
            title="Left Hand Metrics",
            labels=["Rot", "Thumb", "Index", "Middle", "Ring", "Pinky", "F/B", "Side"],
            y_min=0.0,
            y_max=100.0,
            brush="#3aa6ff",
            value_formatter=format_hand_value,
            widget_height=300,
        )
        self.right_widget = MetricsBarWidget(
            title="Right Hand Metrics",
            labels=["Rot", "Thumb", "Index", "Middle", "Ring", "Pinky", "F/B", "Side"],
            y_min=0.0,
            y_max=100.0,
            brush="#66d17a",
            value_formatter=format_hand_value,
            widget_height=300,
        )
        self.hands_splitter.addWidget(self.left_widget)
        self.hands_splitter.addWidget(self.right_widget)
        self.hands_splitter.setStretchFactor(0, 1)
        self.hands_splitter.setStretchFactor(1, 1)
        hands_layout.addWidget(self.hands_splitter)

        self.vertical_splitter.addWidget(top_container)
        self.vertical_splitter.addWidget(hands_container)
        self.vertical_splitter.setStretchFactor(0, 4)
        self.vertical_splitter.setStretchFactor(1, 2)

        self.runtime: VisionRuntime | None = None
        self._start_runtime(camera_index)
        self.source_button.clicked.connect(self._on_change_source_clicked)
        self.invert_checkbox.toggled.connect(self._on_invert_toggled)
        self._on_invert_toggled(self.invert_checkbox.isChecked())
        QTimer.singleShot(0, self._restore_layout)

    def _on_error(self, error: str) -> None:
        self.video_label.set_error_text(f"Runtime error: {error}")

    def _on_frame_ready(self, image, metrics, metrics_valid, head_metrics, diagnostics, fps: float) -> None:
        self.video_label.set_frame(image)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.face_status.setText(f"Face: {diagnostics.get('face', '-')}")
        self.gaze_status.setText(f"Gaze: {diagnostics.get('gaze', '-')}")
        self.left_status.setText(f"Left hand: {diagnostics.get('left_hand', '-')}")
        self.right_status.setText(f"Right hand: {diagnostics.get('right_hand', '-')}")
        self.head_widget.update_metrics(head_metrics)

        left_values = metrics.get("left", [0.0] * 8)
        right_values = metrics.get("right", [0.0] * 8)
        left_valid = metrics_valid.get("left", [False] * 8)
        right_valid = metrics_valid.get("right", [False] * 8)

        if diagnostics.get("left_hand", "lost") != "detected":
            left_valid = [False] * len(left_values)
        if diagnostics.get("right_hand", "lost") != "detected":
            right_valid = [False] * len(right_values)

        self.left_widget.update_metrics(left_values, left_valid)
        self.right_widget.update_metrics(right_values, right_valid)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_runtime()
        save_layout_settings(self._settings_path, self._capture_layout_settings())
        super().closeEvent(event)

    def _start_runtime(self, camera_index: int) -> None:
        self._stop_runtime()
        self.runtime = VisionRuntime(camera_index=camera_index)
        self.runtime.frame_ready.connect(self._on_frame_ready)
        self.runtime.error.connect(self._on_error)
        self.runtime.set_camera_inverted(self.invert_checkbox.isChecked())
        self.source_label.setText(f"Источник: Webcam #{camera_index}")
        self.runtime.start()

    def _stop_runtime(self) -> None:
        if self.runtime is None:
            return
        self.runtime.stop()
        self.runtime = None

    def _on_change_source_clicked(self) -> None:
        dialog = SourceSelectDialog(self, auto_accept_single=False)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._start_runtime(dialog.camera_index)

    def _on_invert_toggled(self, enabled: bool) -> None:
        if self.runtime is not None:
            self.runtime.set_camera_inverted(enabled)

    def _restore_layout(self) -> None:
        self._restore_splitter_sizes(
            self.top_splitter,
            self._layout_settings.top_splitter_sizes,
            [320, 980],
        )
        self._restore_splitter_sizes(
            self.vertical_splitter,
            self._layout_settings.vertical_splitter_sizes,
            [620, 320],
        )
        self._restore_splitter_sizes(
            self.hands_splitter,
            self._layout_settings.hands_splitter_sizes,
            [1, 1],
        )

    def _capture_layout_settings(self) -> LayoutSettings:
        return LayoutSettings(
            window_width=max(800, self.width()),
            window_height=max(600, self.height()),
            top_splitter_sizes=self.top_splitter.sizes(),
            vertical_splitter_sizes=self.vertical_splitter.sizes(),
            hands_splitter_sizes=self.hands_splitter.sizes(),
            invert_camera=self.invert_checkbox.isChecked(),
        )

    @staticmethod
    def _restore_splitter_sizes(splitter: QSplitter, stored: list[int] | None, default: list[int]) -> None:
        if stored and len(stored) == splitter.count() and min(stored) > 0:
            min_size = 120
            adjusted = [max(min_size, value) for value in stored]
            splitter.setSizes(adjusted)
            return
        splitter.setSizes(default)


class AspectRatioVideoLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("Camera stream")
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #090909; color: #aaaaaa; border: 1px solid #2a2a2a; }")

    def set_frame(self, image) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self._refresh()

    def set_error_text(self, text: str) -> None:
        self._source_pixmap = None
        self.clear()
        self.setText(text)

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source_pixmap is None:
            return
        scaled = self._source_pixmap.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


def run_app() -> None:
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    source_dialog: SourceSelectDialog | None = None

    def _handle_terminate(signum, _frame) -> None:  # type: ignore[no-untyped-def]
        if signum not in (signal.SIGINT, signal.SIGTERM):
            return
        if source_dialog is not None and source_dialog.isVisible():
            source_dialog.reject()
        app.closeAllWindows()
        app.quit()

    signal.signal(signal.SIGINT, _handle_terminate)
    signal.signal(signal.SIGTERM, _handle_terminate)

    signal_pump = QTimer()
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start(150)

    source_dialog = SourceSelectDialog(auto_accept_single=True)
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


def _percent_to_deg90(value: float) -> float:
    return (value / 100.0) * 180.0 - 90.0


def format_hand_value(index: int, value: float) -> str:
    if index in (6, 7):
        return f"{_percent_to_deg90(value):+.0f}°"
    return f"{value:.0f}%"
