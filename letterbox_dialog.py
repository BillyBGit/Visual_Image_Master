"""
LetterboxDialog — detect and remove letterbox/pillarbox from images,
then resize to a target square dimension.

Processing pipeline per image:
  1. Sample border colour from the 4 corners.
  2. Walk inward from each edge; trim rows/columns whose maximum per-pixel
     deviation from the border colour is ≤ threshold.
  3. Resize the cropped content to the target square using the chosen mode:
       Fill  — scale so the shorter side = target, then centre-crop to square.
       Fit   — scale so the longer side = target, pad the rest (border colour).
"""

import io
import os

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)


# ── Core processing ───────────────────────────────────────────────────────────


def _detect_crop_box(
    arr: np.ndarray, threshold: int
) -> tuple[int, int, int, int] | None:
    """
    Return (left, top, right, bottom) crop box with letterbox borders removed,
    or None if no letterbox is detected.

    arr  — H×W×3 uint8 numpy array (RGB).
    """
    h, w = arr.shape[:2]
    bc = np.array(
        [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.float32
    ).mean(axis=0)  # [R, G, B] border colour

    # Per-pixel max-channel deviation from border colour
    dev = np.abs(arr.astype(np.float32) - bc).max(axis=2)  # H×W

    content_rows = np.where(dev.max(axis=1) > threshold)[0]
    content_cols = np.where(dev.max(axis=0) > threshold)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return None  # Entire image is uniform — don't crop

    top = int(content_rows[0])
    bottom = int(content_rows[-1]) + 1
    left = int(content_cols[0])
    right = int(content_cols[-1]) + 1

    if top == 0 and bottom == h and left == 0 and right == w:
        return None  # No change needed

    return (left, top, right, bottom)


def _resize_to_square(
    img: Image.Image, size: int, mode: str, pad_colour: tuple
) -> Image.Image:
    """
    Resize a PIL image to an exact size×size square.

    mode="fill"  — scale so the shorter side = size, then centre-crop.
    mode="fit"   — scale so the longer side = size, pad with pad_colour.
    """
    w, h = img.size
    if mode == "fill":
        scale = size / min(w, h)
        nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
        resized = img.resize((nw, nh), Image.LANCZOS)
        x = (nw - size) // 2
        y = (nh - size) // 2
        return resized.crop((x, y, x + size, y + size))
    else:  # fit
        scale = size / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        resized = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), tuple(int(c) for c in pad_colour))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        return canvas


def process_one(
    path: str, threshold: int, size: int, resize_mode: str, overwrite: bool, suffix: str
) -> tuple[bool, str]:
    """
    Remove letterbox from a single image and save the result.
    Returns (success, message).
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)

        box = _detect_crop_box(arr, threshold)
        cropped = img.crop(box) if box else img

        pad_colour = (0, 0, 0)
        if box:
            bc = np.array(
                [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.float32
            ).mean(axis=0)
            pad_colour = tuple(int(c) for c in bc)

        result = _resize_to_square(cropped, size, resize_mode, pad_colour)

        stem, ext = os.path.splitext(path)
        out = path if overwrite else f"{stem}{suffix}{ext}"
        result.save(out, quality=95)
        return True, out
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def preview_result(
    path: str, threshold: int, size: int, resize_mode: str
) -> Image.Image | None:
    """Return the processed PIL image for preview, or None on error."""
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        box = _detect_crop_box(arr, threshold)
        cropped = img.crop(box) if box else img
        bc = (
            np.array(
                [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.float32
            ).mean(axis=0)
            if box
            else np.array([0, 0, 0], dtype=np.float32)
        )
        return _resize_to_square(cropped, size, resize_mode, tuple(int(c) for c in bc))
    except Exception:  # noqa: BLE001
        return None


# ── Worker thread ─────────────────────────────────────────────────────────────


class _Worker(QThread):
    progress = Signal(int, str)  # (n_done, current_filename)
    finished = Signal(int, int)  # (n_ok, n_fail)

    def __init__(self, paths, threshold, size, resize_mode, overwrite, suffix):
        super().__init__()
        self.paths = paths
        self.threshold = threshold
        self.size = size
        self.resize_mode = resize_mode
        self.overwrite = overwrite
        self.suffix = suffix
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok = fail = 0
        for i, path in enumerate(self.paths):
            if self._cancel:
                break
            self.progress.emit(i, os.path.basename(path))
            success, _ = process_one(
                path,
                self.threshold,
                self.size,
                self.resize_mode,
                self.overwrite,
                self.suffix,
            )
            if success:
                ok += 1
            else:
                fail += 1
        self.finished.emit(ok, fail)


# ── PIL image → QPixmap helper ────────────────────────────────────────────────


def _pil_to_pixmap(img: Image.Image) -> QPixmap:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    px = QPixmap()
    px.loadFromData(buf.getvalue())
    return px


# ── Dialog ────────────────────────────────────────────────────────────────────


class LetterboxDialog(QDialog):
    """Remove letterbox/pillarbox borders and resize to a target square."""

    def __init__(self, image_paths: list[str], parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.is_batch = len(image_paths) > 1
        self._worker: _Worker | None = None

        n = len(image_paths)
        title = "Remove Letterbox"
        if self.is_batch:
            title += f" — {n} images"
        self.setWindowTitle(title)
        self.setMinimumWidth(640)

        self._build_ui()
        if not self.is_batch and image_paths:
            self._refresh_preview()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Preview / batch info ──────────────────────────────────────────────
        if self.is_batch:
            info = QLabel(
                f"<b>{len(self.image_paths)} images</b> selected for processing.<br>"
                "Configure settings below and click Apply."
            )
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info.setStyleSheet("color: #aaa; font-size: 13px; padding: 16px;")
            root.addWidget(info)
        else:
            preview_box = QGroupBox("Preview")
            preview_layout = QHBoxLayout(preview_box)

            before_col = QVBoxLayout()
            before_col.addWidget(
                QLabel("Before", alignment=Qt.AlignmentFlag.AlignCenter)
            )
            self._before_label = QLabel()
            self._before_label.setFixedSize(280, 280)
            self._before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._before_label.setStyleSheet(
                "background:#1e1e1e; border:1px solid #555;"
            )
            before_col.addWidget(self._before_label)
            preview_layout.addLayout(before_col)

            after_col = QVBoxLayout()
            after_col.addWidget(QLabel("After", alignment=Qt.AlignmentFlag.AlignCenter))
            self._after_label = QLabel()
            self._after_label.setFixedSize(280, 280)
            self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._after_label.setStyleSheet(
                "background:#1e1e1e; border:1px solid #555;"
            )
            after_col.addWidget(self._after_label)
            preview_layout.addLayout(after_col)

            root.addWidget(preview_box)

        # ── Settings ──────────────────────────────────────────────────────────
        settings_box = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_box)

        # Threshold row
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Detection threshold:"))
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(1, 60)
        self._thresh_slider.setValue(15)
        self._thresh_slider.setTickInterval(5)
        self._thresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._thresh_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._thresh_val = QLabel("15")
        self._thresh_val.setFixedWidth(28)
        self._thresh_slider.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self._thresh_slider)
        thresh_row.addWidget(self._thresh_val)
        settings_layout.addLayout(thresh_row)

        thresh_hint = QLabel(
            "Lower = stricter (only very dark/light borders removed). "
            "Higher = looser (removes near-black or near-white borders too)."
        )
        thresh_hint.setStyleSheet("color: #888; font-size: 10px;")
        thresh_hint.setWordWrap(True)
        settings_layout.addWidget(thresh_hint)

        # Target size row
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Target size:"))
        self._size_group = QButtonGroup(self)
        for sz in (512, 768, 1024):
            rb = QRadioButton(f"{sz} × {sz}")
            self._size_group.addButton(rb, sz)
            size_row.addWidget(rb)
            if sz == 512:
                rb.setChecked(True)
        size_row.addStretch()
        settings_layout.addLayout(size_row)

        # Resize mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Resize mode:"))
        self._mode_group = QButtonGroup(self)
        self._fill_rb = QRadioButton("Fill (scale to cover, centre-crop)")
        self._fit_rb = QRadioButton("Fit (scale to fit, pad with border colour)")
        self._fill_rb.setChecked(True)
        self._mode_group.addButton(self._fill_rb, 0)
        self._mode_group.addButton(self._fit_rb, 1)
        mode_row.addWidget(self._fill_rb)
        mode_row.addWidget(self._fit_rb)
        mode_row.addStretch()
        settings_layout.addLayout(mode_row)

        # Refresh preview button (single-image only)
        if not self.is_batch:
            refresh_btn = QPushButton("Refresh Preview")
            refresh_btn.clicked.connect(self._refresh_preview)
            settings_layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        root.addWidget(settings_box)

        # ── Save mode ─────────────────────────────────────────────────────────
        save_box = QGroupBox("Save Mode")
        save_layout = QHBoxLayout(save_box)
        self._save_group = QButtonGroup(self)
        self._overwrite_rb = QRadioButton("Overwrite original")
        self._copy_rb = QRadioButton("Save as new file  (suffix: _delb)")
        self._overwrite_rb.setChecked(True)
        self._save_group.addButton(self._overwrite_rb, 0)
        self._save_group.addButton(self._copy_rb, 1)
        save_layout.addWidget(self._overwrite_rb)
        save_layout.addWidget(self._copy_rb)
        save_layout.addStretch()
        root.addWidget(save_box)

        # ── Progress bar ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, max(1, len(self.image_paths)))
        self._progress.setVisible(False)
        self._status_label = QLabel()
        self._status_label.setVisible(False)
        root.addWidget(self._progress)
        root.addWidget(self._status_label)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton(
            "Apply" if not self.is_batch else f"Apply to {len(self.image_paths)} images"
        )
        self._apply_btn.setDefault(True)
        self._cancel_btn = QPushButton("Cancel")
        self._apply_btn.clicked.connect(self._apply)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._cancel_btn)
        root.addLayout(btn_row)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_threshold_changed(self, val: int):
        self._thresh_val.setText(str(val))

    def _current_settings(self):
        threshold = self._thresh_slider.value()
        size = self._size_group.checkedId()
        resize_mode = "fill" if self._fill_rb.isChecked() else "fit"
        overwrite = self._overwrite_rb.isChecked()
        return threshold, size, resize_mode, overwrite

    def _refresh_preview(self):
        if not self.image_paths:
            return
        path = self.image_paths[0]
        threshold, size, resize_mode, _ = self._current_settings()

        # Before
        try:
            before_img = Image.open(path).convert("RGB")
            px = _pil_to_pixmap(before_img)
            self._before_label.setPixmap(
                px.scaled(
                    280,
                    280,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception:  # noqa: BLE001
            self._before_label.setText("(load error)")

        # After
        after_img = preview_result(path, threshold, size, resize_mode)
        if after_img:
            px = _pil_to_pixmap(after_img)
            self._after_label.setPixmap(
                px.scaled(
                    280,
                    280,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._after_label.setText("(preview error)")

    def _apply(self):
        threshold, size, resize_mode, overwrite = self._current_settings()

        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setVisible(True)
        self._status_label.setText("Starting…")

        self._worker = _Worker(
            self.image_paths, threshold, size, resize_mode, overwrite, "_delb"
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, n_done: int, filename: str):
        self._progress.setValue(n_done)
        self._status_label.setText(f"Processing {filename}…")

    def _on_finished(self, ok: int, fail: int):
        self._progress.setValue(len(self.image_paths))
        self._status_label.setText(f"Done — {ok} succeeded, {fail} failed.")
        self._apply_btn.setEnabled(True)
        self._worker = None

        if fail == 0:
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Letterbox Removal",
                f"{ok} image(s) processed successfully.\n{fail} failed.",
            )

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        self.reject()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        super().closeEvent(event)
