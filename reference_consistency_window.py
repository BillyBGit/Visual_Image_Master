"""
Reference Folder Consistency Check window - a one-shot audit of a
reference-image folder (the kind used by the Sort menu's reference-folder
similarity sorts) for photos that don't facially match the rest of the
folder, e.g. a mislabeled or wrong-identity photo.
"""

import os

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QProgressBar,
    QApplication,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor

from face_embedding import get_embedder
from preview_window import ImagePreviewWindow

VALID_IMAGE_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".avif",
    ".tiff",
    ".bmp",
)

THUMB_SIZE = 200
BORDER_WIDTH = 6


class ReferenceConsistencyWindow(QMainWindow):
    """Shows every image in a reference folder with its leave-one-out best
    facial match against every OTHER image in that same folder - a photo
    that's mislabeled, wrong-identity, or just a poor/ambiguous crop will
    score noticeably lower than genuine photos of the same person, and
    gets a red border once its score falls below the threshold.

    The threshold is adjustable after the fact without recomputing
    anything, since the (slow) embedding step and the (instant) border
    rendering are kept separate.
    """

    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.setWindowTitle(f"Reference Folder Consistency Check - {folder_path}")
        self.resize(900, 700)

        self.scores = {}  # full_path -> float or None (no detected face)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Flag threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(3)
        # Slightly above FaceEmbedder.SAME_IDENTITY_THRESHOLD (0.363): that
        # constant is calibrated for "are these two photos the same person"
        # in general, but a genuine member of a curated same-person
        # reference folder should usually score well clear of it against
        # the REST of that folder - a photo that only barely clears the
        # generic threshold is still worth a second look.
        self.threshold_spin.setValue(0.40)
        self.threshold_spin.valueChanged.connect(self._apply_threshold)
        top_bar.addWidget(self.threshold_spin)

        self.recompute_btn = QPushButton("Recompute")
        self.recompute_btn.setToolTip(
            "Re-run face detection/embedding (only needed if the folder's "
            "contents changed - adjusting the threshold above doesn't "
            "require this)"
        )
        self.recompute_btn.clicked.connect(self.compute_scores)
        top_bar.addWidget(self.recompute_btn)

        top_bar.addStretch()
        self.status_label = QLabel("")
        top_bar.addWidget(self.status_label)
        layout.addLayout(top_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.grid.itemDoubleClicked.connect(self._open_preview)
        layout.addWidget(self.grid)

        self.compute_scores()

    def _list_images(self):
        try:
            names = sorted(os.listdir(self.folder_path))
        except OSError:
            return []
        return [
            os.path.join(self.folder_path, n)
            for n in names
            if n.lower().endswith(VALID_IMAGE_EXTS)
        ]

    def compute_scores(self):
        """Run (or re-run) face detection/embedding for every image in the
        folder and leave-one-out match each against every other image."""
        paths = self._list_images()
        if not paths:
            self.status_label.setText("No images found in this folder.")
            return

        embedder = get_embedder()
        if not embedder.is_available():
            self.status_label.setText(
                "Face recognition models are not available - see console."
            )
            return

        self.progress_bar.setRange(0, len(paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Detecting faces...")
        QApplication.processEvents()

        all_embeddings = {}
        for i, p in enumerate(paths):
            all_embeddings[p] = embedder.get_face_embeddings(p)
            self.progress_bar.setValue(i + 1)
            if i % 5 == 0:
                QApplication.processEvents()

        self.scores = {}
        for p in paths:
            own_embs = all_embeddings[p]
            if not own_embs:
                self.scores[p] = None
                continue
            others = []
            for other_p, other_embs in all_embeddings.items():
                if other_p != p:
                    others.extend(other_embs)
            if not others:
                self.scores[p] = None
                continue
            per_ref = embedder.best_match_per_reference(own_embs, others)
            self.scores[p] = max(per_ref) if per_ref else None

        self.progress_bar.setVisible(False)
        self._populate_grid(paths)
        self._apply_threshold()

    def _populate_grid(self, paths):
        self.grid.clear()
        for p in paths:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setIcon(QIcon(p))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid.addItem(item)

    def _apply_threshold(self):
        """Re-render borders/tooltips for the current threshold - no
        detection/embedding work, just re-reading already-computed scores,
        so this is instant even for a large reference folder."""
        threshold = self.threshold_spin.value()
        flagged = 0
        for i in range(self.grid.count()):
            item = self.grid.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            score = self.scores.get(path)
            is_flagged = score is None or score < threshold
            if is_flagged:
                flagged += 1
            item.setIcon(QIcon(self._bordered_pixmap(path, is_flagged)))
            if score is None:
                item.setToolTip(f"{os.path.basename(path)}\nNo face detected")
            else:
                item.setToolTip(
                    f"{os.path.basename(path)}\nBest match to rest of folder: "
                    f"{score:.4f}"
                )

        total = self.grid.count()
        self.status_label.setText(
            f"{flagged} of {total} image(s) below threshold ({threshold:.3f})"
        )

    def _bordered_pixmap(self, path, flagged):
        pixmap = QPixmap(path).scaled(
            THUMB_SIZE,
            THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not flagged:
            return pixmap

        canvas = QPixmap(pixmap.size())
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, pixmap)
        pen = QPen(QColor(255, 0, 0), BORDER_WIDTH)
        painter.setPen(pen)
        half = BORDER_WIDTH // 2
        painter.drawRect(
            half, half, canvas.width() - BORDER_WIDTH, canvas.height() - BORDER_WIDTH
        )
        painter.end()
        return canvas

    def _open_preview(self, item):
        paths = [
            self.grid.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.grid.count())
        ]
        current_idx = self.grid.row(item)
        preview = ImagePreviewWindow(paths, current_idx, self)
        preview.show()
