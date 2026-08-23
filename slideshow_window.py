"""
Slideshow — auto-advancing full-window image viewer.

Plays through a list of images (sequential or random order), each shown for
a configurable number of seconds, scaled to fit the window while preserving
aspect ratio.

Keys:
  P       — pause / resume
  ← / →  — previous / next (resets the per-image timer)
  Escape  — close
"""

import os
import random

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QSizePolicy,
    QStatusBar,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap


class SlideshowWindow(QMainWindow):
    """Auto-advancing slideshow viewer for a list of image paths."""

    def __init__(self, image_list, current_index=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slideshow")
        self.resize(1200, 800)

        self.image_list = list(image_list)
        self.order_mode = "sequential"
        self.interval_seconds = 5
        self.paused = False
        self._original_pixmap = None
        self.current_index = 0
        self._was_maximized_before_fullscreen = False

        self._order = self._build_order()
        start_actual = current_index if 0 <= current_index < len(self.image_list) else 0
        self._order_pos = self._order.index(start_actual) if self._order else 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        if self.image_list:
            self._show_image()
            self._restart_timer()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Wrapped in its own widget so the whole control row can be hidden as
        # a unit in fullscreen mode, for a clean, distraction-free view.
        self.controls_widget = QWidget()
        top_row = QHBoxLayout(self.controls_widget)
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(QLabel("Order:"))
        self.seq_radio = QRadioButton("Sequential")
        self.rand_radio = QRadioButton("Random")
        self.seq_radio.setChecked(True)
        order_group = QButtonGroup(self)
        order_group.addButton(self.seq_radio)
        order_group.addButton(self.rand_radio)
        self.seq_radio.toggled.connect(
            lambda checked: self._on_order_changed("sequential") if checked else None
        )
        self.rand_radio.toggled.connect(
            lambda checked: self._on_order_changed("random") if checked else None
        )
        top_row.addWidget(self.seq_radio)
        top_row.addWidget(self.rand_radio)
        top_row.addSpacing(16)

        top_row.addWidget(QLabel("Seconds per image:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 300)
        self.interval_spin.setValue(self.interval_seconds)
        self.interval_spin.valueChanged.connect(self._on_interval_changed)
        top_row.addWidget(self.interval_spin)
        top_row.addSpacing(16)

        prev_btn = QPushButton("◀ Previous")
        prev_btn.clicked.connect(self._go_previous)
        top_row.addWidget(prev_btn)

        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self._toggle_pause)
        top_row.addWidget(self.play_pause_btn)

        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._go_next)
        top_row.addWidget(next_btn)

        fullscreen_btn = QPushButton("Fullscreen")
        fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        top_row.addWidget(fullscreen_btn)

        top_row.addStretch()
        root.addWidget(self.controls_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setMinimumSize(50, 50)
        self.image_label.setStyleSheet("background: #000;")
        root.addWidget(self.image_label, stretch=1)

        self.hint_label = QLabel(
            "F11 = fullscreen  ·  P = pause/resume  ·  ← / → = previous/next  ·  "
            "Esc = exit fullscreen / close"
        )
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("color: #777; font-size: 10px;")
        root.addWidget(self.hint_label)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ── Ordering ──────────────────────────────────────────────────────────

    def _build_order(self):
        order = list(range(len(self.image_list)))
        if self.order_mode == "random":
            random.shuffle(order)
        return order

    def _on_order_changed(self, mode):
        if mode == self.order_mode or not self._order:
            return
        current_actual = self._order[self._order_pos]
        self.order_mode = mode
        self._order = self._build_order()
        self._order_pos = (
            self._order.index(current_actual) if current_actual in self._order else 0
        )
        self._restart_timer()

    def _on_interval_changed(self, value):
        self.interval_seconds = value
        if not self.paused:
            self._timer.start(self.interval_seconds * 1000)

    # ── Playback ──────────────────────────────────────────────────────────

    def _restart_timer(self):
        self._timer.stop()
        if not self.paused and len(self.image_list) > 1:
            self._timer.start(self.interval_seconds * 1000)

    def _advance_pos(self, delta):
        if not self._order:
            return
        n = len(self._order)
        new_pos = self._order_pos + delta
        if new_pos >= n:
            if self.order_mode == "random":
                # Reshuffle each time a full random cycle completes, so it
                # doesn't just repeat the same shuffled order forever.
                self._order = self._build_order()
            new_pos = 0
        elif new_pos < 0:
            new_pos = n - 1
        self._order_pos = new_pos
        self._show_image()

    def _on_timer_tick(self):
        self._advance_pos(1)

    def _go_next(self):
        self._advance_pos(1)
        self._restart_timer()

    def _go_previous(self):
        self._advance_pos(-1)
        self._restart_timer()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.play_pause_btn.setText("Resume" if self.paused else "Pause")
        if self.paused:
            self._timer.stop()
            self.status_bar.showMessage("Paused", 2000)
        else:
            if len(self.image_list) > 1:
                self._timer.start(self.interval_seconds * 1000)
            self.status_bar.showMessage("Resumed", 2000)

    def _toggle_fullscreen(self):
        """Toggle a true windowless fullscreen mode - no title bar/decorations,
        with the control row and status bar hidden for a clean, distraction-
        free view."""
        if self.isFullScreen():
            self.showNormal()
            if self._was_maximized_before_fullscreen:
                self.showMaximized()
            self.controls_widget.setVisible(True)
            self.hint_label.setVisible(True)
            self.status_bar.setVisible(True)
        else:
            self._was_maximized_before_fullscreen = self.isMaximized()
            self.controls_widget.setVisible(False)
            self.hint_label.setVisible(False)
            self.status_bar.setVisible(False)
            self.showFullScreen()
        self._update_scaled_pixmap()

    # ── Display ───────────────────────────────────────────────────────────

    def _show_image(self):
        if not self.image_list:
            return
        self.current_index = self._order[self._order_pos]
        path = self.image_list[self.current_index]
        self._original_pixmap = QPixmap(path)
        self._update_scaled_pixmap()

        n = len(self.image_list)
        self.status_bar.showMessage(
            f"{self._order_pos + 1} / {n}  ·  {os.path.basename(path)}"
        )
        self.setWindowTitle(f"Slideshow — {os.path.basename(path)}")

    def _update_scaled_pixmap(self):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            self.image_label.setText("(could not load image)")
            return
        avail = self.image_label.size()
        if avail.width() > 10 and avail.height() > 10:
            scaled = self._original_pixmap.scaled(
                avail,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    # ── Keyboard ──────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_P:
            self._toggle_pause()
        elif key == Qt.Key.Key_Left:
            self._go_previous()
        elif key == Qt.Key.Key_Right:
            self._go_next()
        elif key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self._toggle_fullscreen()
            else:
                self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
