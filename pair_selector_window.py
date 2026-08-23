"""
Pair Selector — side-by-side image picker with two navigation modes.

Filename mode:  pairs files by name similarity, shows only matched pairs.
                Unmatched files are skipped automatically.
Position mode:  two independent sorted lists with per-pane position spinboxes.
                Either spinbox can be re-entered at any time to re-align.

Keys (both modes):
  1       — copy original (left) to output, advance
  0       — copy processed (right) to output, advance
  Space   — skip, advance
  ← / →  — navigate without copying (review)
  Escape  — close
"""

import os
import shutil

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSplitter,
    QPushButton,
    QFileDialog,
    QStatusBar,
    QApplication,
    QSizePolicy,
    QFrame,
    QSpinBox,
    QRadioButton,
    QStackedWidget,
)
from PySide6.QtCore import Qt, QTimer, QRect, QPoint
from PySide6.QtGui import QPixmap, QKeyEvent, QPainter, QPen, QColor

from filename_similarity import VALID_EXTS, scan_sorted, pair_files, auto_pair_positions


def _scan_sorted(directory, sort_mode="name"):
    return scan_sorted(directory, sort_mode)


def _pair_files(files_a, files_b):
    """Return [(path_a, path_b), ...] matched by filename similarity, in A-order."""
    return [(pa, pb) for pa, pb, _score in pair_files(files_a, files_b)]


def _auto_pair_positions(files_a, files_b):
    return auto_pair_positions(files_a, files_b)


# ── Image pane ────────────────────────────────────────────────────────────────


class ImagePane(QFrame):
    """Single image display pane, with an optional position spinbox row."""

    def __init__(self, header_text, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._pixmap = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.header = QLabel(header_text)
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.header.font()
        f.setBold(True)
        f.setPointSize(11)
        self.header.setFont(f)
        layout.addWidget(self.header)

        # Position row — shown only in position mode
        self._pos_widget = QWidget()
        pos_row = QHBoxLayout(self._pos_widget)
        pos_row.setContentsMargins(0, 0, 0, 0)
        pos_row.addStretch()
        pos_row.addWidget(QLabel("Position:"))
        self.spin = QSpinBox()
        self.spin.setMinimum(1)
        self.spin.setMaximum(1)
        self.spin.setFixedWidth(75)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pos_row.addWidget(self.spin)
        self.count_label = QLabel("/ 0")
        self.count_label.setMinimumWidth(55)
        pos_row.addWidget(self.count_label)
        pos_row.addStretch()
        layout.addWidget(self._pos_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image_label.setMinimumSize(100, 100)
        layout.addWidget(self.image_label, stretch=1)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #ccc; font-size: 13px;")
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.info_label.setCursor(Qt.CursorShape.IBeamCursor)
        layout.addWidget(self.info_label)

    @property
    def pixmap(self):
        return self._pixmap

    def set_position_visible(self, visible):
        self._pos_widget.setVisible(visible)

    def set_file_count(self, n):
        self.spin.setMaximum(max(1, n))
        self.count_label.setText(f"/ {n}")

    def set_image(self, path):
        self._pixmap = QPixmap(path)
        self._fit()
        try:
            size_kb = os.path.getsize(path) // 1024
            from PIL import Image

            with Image.open(path) as img:
                w, h = img.size
            self.info_label.setText(
                f"{os.path.basename(path)}\n{w} × {h}  ·  {size_kb} KB"
            )
        except Exception:
            self.info_label.setText(os.path.basename(path))

    def clear(self):
        self._pixmap = None
        self.image_label.clear()
        self.info_label.clear()

    def highlight(self, on):
        if on:
            self.setStyleSheet(
                "ImagePane { border: 3px solid #00c853;"
                " background: rgba(0,200,83,15); }"
            )
        else:
            self.setStyleSheet("")

    def _fit(self):
        if self._pixmap and not self._pixmap.isNull():
            avail = self.image_label.size()
            if avail.width() > 10 and avail.height() > 10:
                scaled = self._pixmap.scaled(
                    avail,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


# ── Swipe compare overlay ─────────────────────────────────────────────────────


class SwipeCompareWidget(QWidget):
    """
    Overlay compare view: image A and image B are drawn on top of each other,
    scaled to the same rect. Left of the divider shows A, right shows B.
    Click-drag anywhere (or drag the handle) to move the divider.
    """

    HANDLE_RADIUS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._pixmap_a = None
        self._pixmap_b = None
        self._ratio = 0.5
        self._dragging = False

    def set_images(self, pixmap_a, pixmap_b):
        self._pixmap_a = pixmap_a
        self._pixmap_b = pixmap_b
        self.update()

    def clear(self):
        self._pixmap_a = None
        self._pixmap_b = None
        self.update()

    def _target_rect(self):
        """Largest centered rect fitting the reference image's aspect ratio."""
        ref = (
            self._pixmap_a
            if self._pixmap_a and not self._pixmap_a.isNull()
            else self._pixmap_b
        )
        if not ref or ref.isNull():
            return QRect()
        avail_w, avail_h = self.width(), self.height()
        if avail_w < 10 or avail_h < 10:
            return QRect()
        img_w, img_h = ref.width(), ref.height()
        if img_w <= 0 or img_h <= 0:
            return QRect()
        scale = min(avail_w / img_w, avail_h / img_h)
        w, h = int(img_w * scale), int(img_h * scale)
        x = (avail_w - w) // 2
        y = (avail_h - h) // 2
        return QRect(x, y, w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self._target_rect()
        if rect.isEmpty():
            painter.setPen(QColor("#777"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load both an original and processed image to compare",
            )
            return

        divider_x = rect.left() + round(rect.width() * self._ratio)

        if self._pixmap_b and not self._pixmap_b.isNull():
            painter.drawPixmap(rect, self._pixmap_b)
        if self._pixmap_a and not self._pixmap_a.isNull():
            painter.save()
            painter.setClipRect(
                QRect(rect.left(), rect.top(), divider_x - rect.left(), rect.height())
            )
            painter.drawPixmap(rect, self._pixmap_a)
            painter.restore()

        pen = QPen(QColor("#00c853"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(divider_x, rect.top(), divider_x, rect.bottom())

        painter.setBrush(QColor("#00c853"))
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawEllipse(
            QPoint(divider_x, rect.top() + rect.height() // 2),
            self.HANDLE_RADIUS,
            self.HANDLE_RADIUS,
        )

        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect.left() + 8, rect.top() + 18, "A — Original")
        b_text = "Processed — B"
        b_width = painter.fontMetrics().horizontalAdvance(b_text)
        painter.drawText(rect.right() - b_width - 8, rect.top() + 18, b_text)

    def _update_ratio_from_x(self, x):
        rect = self._target_rect()
        if rect.isEmpty() or rect.width() == 0:
            return
        ratio = (x - rect.left()) / rect.width()
        self._ratio = min(1.0, max(0.0, ratio))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_ratio_from_x(event.position().x())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_ratio_from_x(event.position().x())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def mouseDoubleClickEvent(self, event):
        self._ratio = 0.5
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


# ── Main window ───────────────────────────────────────────────────────────────


class PairSelectorWindow(QMainWindow):
    """Side-by-side image picker with filename-pair and position modes."""

    def __init__(self, parent=None, dir_a=None, dir_b=None, files_a=None):
        super().__init__(parent)
        self.setWindowTitle("Pair Selector")
        self.resize(1400, 820)

        # Shared
        self._files_a = []
        self._files_b = []
        self._sort_mode = "name"
        self._dir_a = ""
        self._dir_b = ""
        self._dir_out = ""
        self._mode = "filename"  # "filename" or "position"

        # Filename-mode state
        self._pairs = []  # [(path_a, path_b), ...]
        self._pair_idx = 0
        self._pair_decisions = {}  # pair_idx -> "a"|"b"|"skip"

        # Position-mode state
        self._pos_a = 0
        self._pos_b = 0
        self._decisions = {}  # (pos_a, pos_b) -> "a"|"b"|"skip"

        self._build_ui()
        self._apply_mode()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Pre-populate from caller-supplied directories / file list.
        # files_a (specific selected files) takes priority over dir_a.
        if files_a:
            valid = [f for f in files_a if os.path.splitext(f)[1].lower() in VALID_EXTS]
            if valid:
                self._files_a = valid
                self._dir_a = os.path.dirname(valid[0])
                self._btn_a.setText(f"Original: {len(valid)} selected file(s)")
                self._pane_a.set_file_count(len(valid))
                self._rebuild_pairs()
                self._show_current()
        elif dir_a and os.path.isdir(dir_a):
            self._dir_a = dir_a
            self._btn_a.setText(f"Original: …/{os.path.basename(dir_a)}")
            self._reload_a()

        if dir_b and os.path.isdir(dir_b):
            self._dir_b = dir_b
            self._btn_b.setText(f"Processed: …/{os.path.basename(dir_b)}")
            self._reload_b()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Top bar ──────────────────────────────────────────────────────────
        top_row = QHBoxLayout()

        self._btn_a = QPushButton("Original folder…")
        self._btn_a.clicked.connect(self._pick_dir_a)
        self._btn_b = QPushButton("Processed folder…")
        self._btn_b.clicked.connect(self._pick_dir_b)
        self._btn_out = QPushButton("Output folder…")
        self._btn_out.clicked.connect(self._pick_dir_out)
        self._btn_copy_caption = QPushButton("Copy Caption →")
        self._btn_copy_caption.setToolTip(
            "Copy the .txt caption from the LHS image to the RHS folder,\n"
            "renamed to match the RHS image filename."
        )
        self._btn_copy_caption.clicked.connect(self._copy_caption_to_rhs)
        top_row.addWidget(self._btn_a)
        top_row.addWidget(self._btn_b)
        top_row.addWidget(self._btn_out)
        top_row.addWidget(self._btn_copy_caption)
        top_row.addStretch()

        top_row.addWidget(QLabel("Mode:"))
        self._mode_filename = QRadioButton("Filename pairs")
        self._mode_position = QRadioButton("Position")
        self._mode_filename.setChecked(True)
        self._mode_filename.toggled.connect(
            lambda checked: self._set_mode("filename") if checked else None
        )
        self._mode_position.toggled.connect(
            lambda checked: self._set_mode("position") if checked else None
        )
        top_row.addWidget(self._mode_filename)
        top_row.addWidget(self._mode_position)
        top_row.addSpacing(16)

        top_row.addWidget(QLabel("Sort:"))
        self._sort_name = QRadioButton("Name")
        self._sort_date = QRadioButton("Date")
        self._sort_name.setChecked(True)
        self._sort_name.toggled.connect(
            lambda checked: self._set_sort("name") if checked else None
        )
        self._sort_date.toggled.connect(
            lambda checked: self._set_sort("date") if checked else None
        )
        top_row.addWidget(self._sort_name)
        top_row.addWidget(self._sort_date)
        top_row.addSpacing(12)

        self._btn_auto = QPushButton("Auto-align")
        self._btn_auto.setToolTip(
            "Detect starting positions automatically from filename similarity"
        )
        self._btn_auto.clicked.connect(self._auto_align)
        top_row.addWidget(self._btn_auto)

        top_row.addSpacing(12)
        self._btn_compare = QPushButton("🔀 Swipe Compare")
        self._btn_compare.setCheckable(True)
        self._btn_compare.setToolTip(
            "Overlay both images with a draggable vertical divider —\n"
            "left of the line shows the original, right shows the processed image."
        )
        self._btn_compare.toggled.connect(self._toggle_compare_view)
        top_row.addWidget(self._btn_compare)

        root.addLayout(top_row)

        # ── Image splitter ───────────────────────────────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._pane_a = ImagePane("◀  Original   [1]")
        self._pane_b = ImagePane("Processed  ▶  [0]")
        self._splitter.addWidget(self._pane_a)
        self._splitter.addWidget(self._pane_b)
        self._splitter.setSizes([700, 700])

        self._swipe = SwipeCompareWidget()

        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._splitter)
        self._view_stack.addWidget(self._swipe)
        root.addWidget(self._view_stack, stretch=1)

        # Debounced spinbox connections — avoids loading intermediate images
        # while the user is still typing a multi-digit position number.
        self._debounce_a = QTimer()
        self._debounce_a.setSingleShot(True)
        self._debounce_a.setInterval(250)
        self._debounce_a.timeout.connect(self._apply_spin_a)

        self._debounce_b = QTimer()
        self._debounce_b.setSingleShot(True)
        self._debounce_b.setInterval(250)
        self._debounce_b.timeout.connect(self._apply_spin_b)

        self._pane_a.spin.valueChanged.connect(lambda _: self._debounce_a.start())
        self._pane_b.spin.valueChanged.connect(lambda _: self._debounce_b.start())
        self._pane_a.spin.editingFinished.connect(self.setFocus)
        self._pane_b.spin.editingFinished.connect(self.setFocus)

        # ── Hint bar ─────────────────────────────────────────────────────────
        hint = QLabel(
            "1 = copy original  ·  0 = copy processed  ·"
            "  Space = skip  ·  ← / → = navigate without copying  ·"
            "  🔀 Swipe Compare: drag the line, double-click to re-center"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #777; font-size: 10px;")
        root.addWidget(hint)

        self.setStatusBar(QStatusBar())

    # ── Mode management ───────────────────────────────────────────────────────

    def _set_mode(self, mode):
        if mode == self._mode:
            return
        self._mode = mode
        self._apply_mode()
        self._show_current()

    def _apply_mode(self):
        """Show/hide controls that are specific to one mode."""
        pos_mode = self._mode == "position"
        self._pane_a.set_position_visible(pos_mode)
        self._pane_b.set_position_visible(pos_mode)
        self._btn_auto.setVisible(pos_mode)

    # ── Compare view ──────────────────────────────────────────────────────────

    def _toggle_compare_view(self, checked):
        self._view_stack.setCurrentWidget(self._swipe if checked else self._splitter)
        if checked:
            self._sync_swipe_images()

    def _sync_swipe_images(self):
        self._swipe.set_images(self._pane_a.pixmap, self._pane_b.pixmap)

    # ── Directory pickers ─────────────────────────────────────────────────────

    def _pick_dir_a(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Original Folder", self._dir_a
        )
        if d:
            self._dir_a = d
            self._btn_a.setText(f"Original: …/{os.path.basename(d)}")
            self._reload_a()

    def _pick_dir_b(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Processed Folder", self._dir_b
        )
        if d:
            self._dir_b = d
            self._btn_b.setText(f"Processed: …/{os.path.basename(d)}")
            self._reload_b()

    def _pick_dir_out(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self._dir_out
        )
        if d:
            self._dir_out = d
            self._btn_out.setText(f"Output: …/{os.path.basename(d)}")

    def _copy_caption_to_rhs(self):
        """Copy the LHS .txt caption to the RHS folder, named after the RHS image."""
        from PySide6.QtWidgets import QMessageBox

        if self._mode == "filename":
            if not self._pairs or self._pair_idx >= len(self._pairs):
                self.statusBar().showMessage(
                    "No current pair to copy caption for.", 4000
                )
                return
            path_a, path_b = self._pairs[self._pair_idx]
        else:
            if not self._files_a or not self._files_b:
                self.statusBar().showMessage("No files loaded.", 4000)
                return
            if self._pos_a >= len(self._files_a) or self._pos_b >= len(self._files_b):
                self.statusBar().showMessage("Position out of range.", 4000)
                return
            path_a = self._files_a[self._pos_a]
            path_b = self._files_b[self._pos_b]

        src_txt = os.path.splitext(path_a)[0] + ".txt"
        if not os.path.exists(src_txt):
            self.statusBar().showMessage(
                f"No caption file alongside {os.path.basename(path_a)}", 4000
            )
            return

        dst_txt = os.path.splitext(path_b)[0] + ".txt"

        if os.path.exists(dst_txt):
            reply = QMessageBox.question(
                self,
                "Overwrite Caption?",
                f"{os.path.basename(dst_txt)} already exists in the RHS folder.\n"
                "Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            shutil.copy2(src_txt, dst_txt)
            self.statusBar().showMessage(
                f"Caption copied → {os.path.basename(dst_txt)}", 4000
            )
        except Exception as e:
            self.statusBar().showMessage(f"Copy failed: {e}", 5000)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _reload_a(self):
        self._files_a = _scan_sorted(self._dir_a, self._sort_mode)
        self._pos_a = 0
        self._pane_a.set_file_count(len(self._files_a))
        self._rebuild_pairs()
        self._show_current()

    def _reload_b(self):
        self._files_b = _scan_sorted(self._dir_b, self._sort_mode)
        self._pos_b = 0
        self._pane_b.set_file_count(len(self._files_b))
        self._rebuild_pairs()
        self._show_current()

    def _rebuild_pairs(self):
        self._pairs = _pair_files(self._files_a, self._files_b)
        self._pair_idx = 0
        self._pair_decisions = {}

    def _set_sort(self, mode):
        if mode == self._sort_mode:
            return
        self._sort_mode = mode
        if self._dir_a:
            cur = self._files_a[self._pos_a] if self._files_a else None
            self._files_a = _scan_sorted(self._dir_a, mode)
            self._pos_a = self._files_a.index(cur) if cur in self._files_a else 0
            self._pane_a.set_file_count(len(self._files_a))
        if self._dir_b:
            cur = self._files_b[self._pos_b] if self._files_b else None
            self._files_b = _scan_sorted(self._dir_b, mode)
            self._pos_b = self._files_b.index(cur) if cur in self._files_b else 0
            self._pane_b.set_file_count(len(self._files_b))
        self._rebuild_pairs()
        self._show_current()

    # ── Auto-align (position mode only) ──────────────────────────────────────

    def _auto_align(self):
        if not self._files_a or not self._files_b:
            self.statusBar().showMessage("Load both folders before using Auto-align.")
            return
        pa, pb = _auto_pair_positions(self._files_a, self._files_b)
        self._pos_a = pa
        self._pos_b = pb
        self._show_current()
        self.statusBar().showMessage(
            f"Auto-aligned: LHS position {pa + 1}, RHS position {pb + 1}"
        )

    # ── Spinbox handlers (position mode, debounced) ───────────────────────────

    def _apply_spin_a(self):
        self._pos_a = self._pane_a.spin.value() - 1
        if 0 <= self._pos_a < len(self._files_a):
            self._pane_a.set_image(self._files_a[self._pos_a])
            decision = self._decisions.get((self._pos_a, self._pos_b))
            self._pane_a.highlight(decision == "a")
        else:
            self._pane_a.clear()
        self._sync_swipe_images()
        self._update_status()
        # Return focus so 1/0/Space keystrokes go to the window, not the spinbox
        self.setFocus()

    def _apply_spin_b(self):
        self._pos_b = self._pane_b.spin.value() - 1
        if 0 <= self._pos_b < len(self._files_b):
            self._pane_b.set_image(self._files_b[self._pos_b])
            decision = self._decisions.get((self._pos_a, self._pos_b))
            self._pane_b.highlight(decision == "b")
        else:
            self._pane_b.clear()
        self._sync_swipe_images()
        self._update_status()
        self.setFocus()

    # ── Display ───────────────────────────────────────────────────────────────

    def _show_current(self):
        if self._mode == "filename":
            self._show_current_pair()
        else:
            self._show_current_pos()

    def _show_current_pair(self):
        if not self._pairs:
            self._pane_a.clear()
            self._pane_b.clear()
            self._sync_swipe_images()
            self._update_status()
            return
        if self._pair_idx >= len(self._pairs):
            self._pair_idx = len(self._pairs) - 1
        pa, pb = self._pairs[self._pair_idx]
        self._pane_a.set_image(pa)
        self._pane_b.set_image(pb)
        decision = self._pair_decisions.get(self._pair_idx)
        self._pane_a.highlight(decision == "a")
        self._pane_b.highlight(decision == "b")
        self._sync_swipe_images()
        self._update_status()

    def _show_current_pos(self):
        if 0 <= self._pos_a < len(self._files_a):
            self._pane_a.set_image(self._files_a[self._pos_a])
        else:
            self._pane_a.clear()

        if 0 <= self._pos_b < len(self._files_b):
            self._pane_b.set_image(self._files_b[self._pos_b])
        else:
            self._pane_b.clear()

        self._pane_a.spin.blockSignals(True)
        self._pane_a.spin.setValue(self._pos_a + 1)
        self._pane_a.spin.blockSignals(False)

        self._pane_b.spin.blockSignals(True)
        self._pane_b.spin.setValue(self._pos_b + 1)
        self._pane_b.spin.blockSignals(False)

        decision = self._decisions.get((self._pos_a, self._pos_b))
        self._pane_a.highlight(decision == "a")
        self._pane_b.highlight(decision == "b")
        self._sync_swipe_images()
        self._update_status()

    def _update_status(self):
        if self._mode == "filename":
            n = len(self._pairs)
            if n == 0:
                na, nb = len(self._files_a), len(self._files_b)
                self.statusBar().showMessage(
                    f"No filename matches found  ·  {na} original, {nb} processed"
                )
                return
            copied = sum(1 for v in self._pair_decisions.values() if v != "skip")
            skipped = sum(1 for v in self._pair_decisions.values() if v == "skip")
            self.statusBar().showMessage(
                f"Pair {self._pair_idx + 1} / {n}"
                f"  ·  {copied} copied  ·  {skipped} skipped"
            )
        else:
            na, nb = len(self._files_a), len(self._files_b)
            copied = sum(1 for v in self._decisions.values() if v != "skip")
            skipped = sum(1 for v in self._decisions.values() if v == "skip")
            self.statusBar().showMessage(
                f"LHS {self._pos_a + 1}/{na}  ·  RHS {self._pos_b + 1}/{nb}"
                f"  ·  {copied} copied  ·  {skipped} skipped"
            )

    # ── Selection actions ─────────────────────────────────────────────────────

    def _pick(self, side):
        if not self._dir_out:
            self.statusBar().showMessage("Please select an output folder first.")
            return

        if self._mode == "filename":
            if not self._pairs or self._pair_idx >= len(self._pairs):
                return
            pa, pb = self._pairs[self._pair_idx]
            src = pa if side == "a" else pb
        else:
            if not self._files_a or not self._files_b:
                return
            if self._pos_a >= len(self._files_a) or self._pos_b >= len(self._files_b):
                return
            src = (
                self._files_a[self._pos_a]
                if side == "a"
                else self._files_b[self._pos_b]
            )

        dest = self._unique_dest(src)
        try:
            os.makedirs(self._dir_out, exist_ok=True)
            shutil.copy2(src, dest)
        except Exception as e:
            self.statusBar().showMessage(f"Copy failed: {e}")
            return

        if self._mode == "filename":
            self._pair_decisions[self._pair_idx] = side
        else:
            self._decisions[(self._pos_a, self._pos_b)] = side

        self._pane_a.highlight(side == "a")
        self._pane_b.highlight(side == "b")
        QApplication.processEvents()
        self._advance()

    def _skip(self):
        if self._mode == "filename":
            if not self._pairs:
                return
            self._pair_decisions[self._pair_idx] = "skip"
        else:
            if not self._files_a and not self._files_b:
                return
            self._decisions[(self._pos_a, self._pos_b)] = "skip"
        self._advance()

    def _advance(self):
        # Cancel any in-flight spinbox debounces — stale timer callbacks must
        # not overwrite positions that _advance() is about to set.
        self._debounce_a.stop()
        self._debounce_b.stop()
        if self._mode == "filename":
            next_idx = self._pair_idx + 1
            if next_idx >= len(self._pairs):
                copied = sum(1 for v in self._pair_decisions.values() if v != "skip")
                skipped = sum(1 for v in self._pair_decisions.values() if v == "skip")
                self.statusBar().showMessage(
                    f"All {len(self._pairs)} pairs processed"
                    f" — {copied} copied, {skipped} skipped."
                )
                return
            self._pair_idx = next_idx
        else:
            new_a = self._pos_a + 1
            new_b = self._pos_b + 1
            if new_a >= len(self._files_a) or new_b >= len(self._files_b):
                copied = sum(1 for v in self._decisions.values() if v != "skip")
                skipped = sum(1 for v in self._decisions.values() if v == "skip")
                self.statusBar().showMessage(
                    f"End of list — {copied} copied, {skipped} skipped."
                )
                return
            self._pos_a = new_a
            self._pos_b = new_b
        self._show_current()

    def _unique_dest(self, src):
        name, ext = os.path.splitext(os.path.basename(src))
        dest = os.path.join(self._dir_out, f"{name}{ext}")
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(self._dir_out, f"{name}_{n}{ext}")
            n += 1
        return dest

    # ── Navigation (review without copying) ───────────────────────────────────

    def _go_prev(self):
        if self._mode == "filename":
            if self._pair_idx > 0:
                self._pair_idx -= 1
                self._show_current()
        else:
            if self._pos_a > 0 and self._pos_b > 0:
                self._pos_a -= 1
                self._pos_b -= 1
                self._show_current()

    def _go_next(self):
        if self._mode == "filename":
            if self._pair_idx < len(self._pairs) - 1:
                self._pair_idx += 1
                self._show_current()
        else:
            if (
                self._pos_a < len(self._files_a) - 1
                and self._pos_b < len(self._files_b) - 1
            ):
                self._pos_a += 1
                self._pos_b += 1
                self._show_current()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_1:
            self._pick("a")
        elif key == Qt.Key.Key_0:
            self._pick("b")
        elif key == Qt.Key.Key_Space:
            self._skip()
        elif key == Qt.Key.Key_Left:
            self._go_prev()
        elif key == Qt.Key.Key_Right:
            self._go_next()
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = PairSelectorWindow()
    win.show()
    sys.exit(app.exec())
