"""
JoinImagesDialog — arrange 2-6 selected images on a grid (drag tiles into
place, similar to an OS display-arrangement dialog) and stitch them into one
combined image. Typical use: reassembling portrait crops that were split
from one larger landscape photo.
"""

import os

import cv2

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from image_join import stitch_grid
from preview_window import ImagePreviewWindow
from utils import find_main_window

_CELL = 150
_PAD = 8
_GAP = 14
_MARGIN = 20
_TRAY_GAP = 34


class _Tile(QGraphicsPixmapItem):
    """A draggable thumbnail representing one source image."""

    def __init__(self, path, canvas):
        pixmap = QPixmap(path)
        avail = _CELL - 2 * _PAD
        scaled = pixmap.scaled(
            avail,
            avail,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().__init__(scaled)
        self.path = path
        self.canvas = canvas
        self.setOffset((_CELL - scaled.width()) / 2, (_CELL - scaled.height()) / 2)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(1)
        self.setToolTip(os.path.basename(path))
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas._on_tile_dropped(self)


class ArrangementCanvas(QGraphicsView):
    """Grid + tray of draggable tiles. Tiles snap to the nearest slot on drop."""

    layout_changed = Signal()

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.rows = 0
        self.cols = 0
        self.slots = {}
        self.grid_keys = []
        self.tile_slot = {}
        self.slot_occupant = {}
        self.tiles = []

    def set_grid(self, rows, cols):
        self._scene.clear()
        self.rows, self.cols = rows, cols
        self.slots = {}
        self.grid_keys = []
        self.tile_slot = {}
        self.slot_occupant = {}
        self.tiles = []

        for r in range(rows):
            for c in range(cols):
                x = _MARGIN + c * (_CELL + _GAP)
                y = _MARGIN + r * (_CELL + _GAP)
                key = ("grid", r, c)
                self.slots[key] = QPointF(x, y)
                self.grid_keys.append(key)
                self.slot_occupant[key] = None
                rect = self._scene.addRect(
                    x, y, _CELL, _CELL, QPen(QColor("#666"), 1, Qt.PenStyle.DashLine)
                )
                rect.setZValue(-1)

        n = len(self.image_paths)
        grid_bottom = _MARGIN + rows * (_CELL + _GAP)
        tray_y = grid_bottom + _TRAY_GAP
        for i in range(n):
            x = _MARGIN + i * (_CELL + _GAP)
            key = ("tray", i)
            self.slots[key] = QPointF(x, tray_y)
            self.slot_occupant[key] = None
            rect = self._scene.addRect(
                x, tray_y, _CELL, _CELL, QPen(QColor("#444"), 1, Qt.PenStyle.SolidLine)
            )
            rect.setZValue(-1)

        for i, path in enumerate(self.image_paths):
            tile = _Tile(path, self)
            key = ("tray", i)
            tile.setPos(self.slots[key])
            self.tile_slot[tile] = key
            self.slot_occupant[key] = tile
            self._scene.addItem(tile)
            self.tiles.append(tile)

        total_w = 2 * _MARGIN + max(cols, n, 1) * (_CELL + _GAP)
        total_h = tray_y + _CELL + _MARGIN
        self._scene.setSceneRect(0, 0, total_w, total_h)
        self.layout_changed.emit()

    def _on_tile_dropped(self, tile):
        cur = tile.pos()
        best_key, best_dist = None, None
        for key, pos in self.slots.items():
            d = (pos.x() - cur.x()) ** 2 + (pos.y() - cur.y()) ** 2
            if best_dist is None or d < best_dist:
                best_dist, best_key = d, key

        old_key = self.tile_slot[tile]
        if best_key == old_key:
            tile.setPos(self.slots[old_key])
            return

        occupant = self.slot_occupant.get(best_key)
        self.slot_occupant[best_key] = tile
        self.tile_slot[tile] = best_key
        tile.setPos(self.slots[best_key])

        if occupant is not None and occupant is not tile:
            self.slot_occupant[old_key] = occupant
            self.tile_slot[occupant] = old_key
            occupant.setPos(self.slots[old_key])
        else:
            self.slot_occupant[old_key] = None

        self.layout_changed.emit()

    def filled_count(self):
        return sum(1 for k in self.grid_keys if self.slot_occupant.get(k) is not None)

    def get_grid_paths(self):
        """Return rows of paths (top-to-bottom, left-to-right) if every grid
        cell is filled, else None."""
        if not self.grid_keys:
            return None
        grid = [[None] * self.cols for _ in range(self.rows)]
        for key in self.grid_keys:
            _, r, c = key
            occupant = self.slot_occupant.get(key)
            if occupant is None:
                return None
            grid[r][c] = occupant.path
        return grid

    # ── Context menu: move/copy/delete the source files behind selected tiles ──

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if not isinstance(item, _Tile):
            return

        if not item.isSelected():
            self._scene.clearSelection()
            item.setSelected(True)

        menu = QMenu(self)
        move_act = menu.addAction("Move to Selected")
        copy_act = menu.addAction("Copy to Selected")
        menu.addSeparator()
        delete_act = menu.addAction("Delete")

        chosen = menu.exec(event.globalPos())
        if chosen == move_act:
            self._transfer_selected_tiles(is_copy=False)
        elif chosen == copy_act:
            self._transfer_selected_tiles(is_copy=True)
        elif chosen == delete_act:
            self._delete_selected_tiles()

    def _selected_tiles(self):
        return [it for it in self._scene.selectedItems() if isinstance(it, _Tile)]

    def _get_selected_dest_folder(self, main_win):
        current_index = main_win.dest_tree.currentIndex()
        if current_index.isValid():
            return main_win.dest_model.filePath(
                main_win.dest_proxy.mapToSource(current_index)
            )
        return main_win.dest_root_path

    def _transfer_selected_tiles(self, is_copy):
        tiles = self._selected_tiles()
        if not tiles:
            return

        main_win = find_main_window()
        if not main_win:
            return

        dest_folder = self._get_selected_dest_folder(main_win)
        if not dest_folder:
            QMessageBox.warning(
                self, "No Destination", "Please select a destination folder first."
            )
            return

        if hasattr(main_win, "start_operation_batch"):
            main_win.start_operation_batch("copy" if is_copy else "move")

        transferred_paths = []
        failed = 0
        for tile in tiles:
            try:
                main_win.smart_transfer(tile.path, dest_folder, is_copy)
                if not is_copy:
                    transferred_paths.append(tile.path)
            except Exception as e:  # noqa: BLE001
                failed += 1
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not transfer {os.path.basename(tile.path)}:\n{e}",
                )

        if not is_copy and transferred_paths:
            self._remove_tiles_by_path(transferred_paths)
            if hasattr(main_win, "remove_thumbnails_by_path"):
                main_win.remove_thumbnails_by_path(transferred_paths)

        operation = "Copied" if is_copy else "Moved"
        success = len(tiles) - failed
        if success > 0 and hasattr(main_win, "status_bar"):
            main_win.status_bar.showMessage(
                f"{operation} {success} file(s) to {os.path.basename(dest_folder)}",
                3000,
            )

    def _delete_selected_tiles(self):
        tiles = self._selected_tiles()
        if not tiles:
            return

        main_win = find_main_window()
        should_confirm = True
        if main_win and hasattr(main_win, "conf_delete_act"):
            should_confirm = main_win.conf_delete_act.isChecked()

        if should_confirm:
            confirm = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete {len(tiles)} file(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        if main_win and hasattr(main_win, "delete_undo_manager"):
            main_win.delete_undo_manager.start_new_deletion()

        deleted_paths = []
        for tile in tiles:
            path = tile.path
            try:
                if os.path.exists(path):
                    if main_win and hasattr(main_win, "delete_undo_manager"):
                        main_win.delete_undo_manager.store_for_undo(path)
                    else:
                        os.remove(path)
                deleted_paths.append(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Error", f"Could not delete {path}:\n{e}")

        if not deleted_paths:
            return

        self._remove_tiles_by_path(deleted_paths)

        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, ImagePreviewWindow):
                widget.handle_external_deletion(deleted_paths)

        if main_win and hasattr(main_win, "remove_thumbnails_by_path"):
            main_win.remove_thumbnails_by_path(deleted_paths)
        if main_win and hasattr(main_win, "status_bar"):
            main_win.status_bar.showMessage(
                f"Deleted {len(deleted_paths)} file(s) - Ctrl+Z to undo", 3000
            )

    def _remove_tiles_by_path(self, paths):
        """Drop tiles for files that were just moved/deleted so a stale path
        can't be read again by a later Join."""
        path_set = set(paths)
        for tile in [t for t in self.tiles if t.path in path_set]:
            slot_key = self.tile_slot.pop(tile, None)
            if slot_key is not None:
                self.slot_occupant[slot_key] = None
            self._scene.removeItem(tile)
            self.tiles.remove(tile)
            self.image_paths = [p for p in self.image_paths if p != tile.path]
        self.layout_changed.emit()


class JoinImagesDialog(QDialog):
    """Arrange 2-6 images on a grid and stitch them into one output image."""

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self._invalid = not (2 <= len(self.image_paths) <= 6)
        if self._invalid:
            self.setWindowTitle("Join Images")
            QTimer.singleShot(0, self._reject_invalid)
            return

        self.setWindowTitle(f"Join Images — {len(self.image_paths)} images")
        self.setMinimumSize(700, 560)
        self._build_ui()

    def _reject_invalid(self):
        QMessageBox.warning(
            self, "Join Images", "Select between 2 and 6 images to join."
        )
        self.reject()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        info = QLabel(
            "Drag each thumbnail into the grid to match how the pieces fit "
            "together, then click Join and Save."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa;")
        root.addWidget(info)

        grid_size_row = QHBoxLayout()
        grid_size_row.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 6)
        grid_size_row.addWidget(self.rows_spin)
        grid_size_row.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 6)
        grid_size_row.addWidget(self.cols_spin)
        grid_size_row.addStretch()
        root.addLayout(grid_size_row)

        self.canvas = ArrangementCanvas(self.image_paths, self)
        self.canvas.setMinimumHeight(360)
        self.canvas.layout_changed.connect(self._update_status)
        root.addWidget(self.canvas)

        self.rows_spin.valueChanged.connect(self._on_grid_size_changed)
        self.cols_spin.valueChanged.connect(self._on_grid_size_changed)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #ccc;")
        root.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.join_btn = QPushButton("Join and Save...")
        self.join_btn.setDefault(True)
        self.join_btn.clicked.connect(self._on_join)
        cancel_btn = QPushButton("Close")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.join_btn)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

        n = len(self.image_paths)
        self.rows_spin.blockSignals(True)
        self.cols_spin.blockSignals(True)
        self.rows_spin.setValue(1)
        self.cols_spin.setValue(min(n, 6))
        self.rows_spin.blockSignals(False)
        self.cols_spin.blockSignals(False)

        self.canvas.set_grid(1, min(n, 6))
        self._update_status()

    def _on_grid_size_changed(self):
        self.canvas.set_grid(self.rows_spin.value(), self.cols_spin.value())
        self._update_status()

    def _update_status(self):
        filled = self.canvas.filled_count()
        total = self.rows_spin.value() * self.cols_spin.value()
        n = len(self.canvas.image_paths)
        ok = total == n and filled == total
        self.join_btn.setEnabled(ok)
        if total != n:
            self.status_label.setText(
                f"Grid has {total} cells but {n} images are selected — "
                "adjust rows/cols so they match."
            )
        else:
            self.status_label.setText(f"{filled} of {total} cells filled.")

    def _on_join(self):
        grid_paths = self.canvas.get_grid_paths()
        if grid_paths is None:
            QMessageBox.warning(
                self, "Join Images", "Fill every grid cell before joining."
            )
            return

        grid_imgs = []
        for row in grid_paths:
            imgs = []
            for path in row:
                img = cv2.imread(path)
                if img is None:
                    QMessageBox.warning(
                        self, "Join Images", f"Could not read image:\n{path}"
                    )
                    return
                imgs.append(img)
            grid_imgs.append(imgs)

        try:
            result = stitch_grid(grid_imgs)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Join Images", f"Failed to join images: {e}")
            return

        first_source = self.canvas.image_paths[0] if self.canvas.image_paths else None
        first_dir = (
            os.path.dirname(first_source) if first_source else os.path.expanduser("~")
        )
        default_path = os.path.join(first_dir or ".", "joined.png")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Joined Image", default_path, "PNG Image (*.png)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".png"):
            out_path += ".png"

        if not cv2.imwrite(out_path, result):
            QMessageBox.warning(
                self, "Join Images", f"Failed to save image to:\n{out_path}"
            )
            return

        QMessageBox.information(
            self, "Join Images", f"Saved joined image to:\n{out_path}"
        )

        # Keep the window open so the user can move/copy/delete the source
        # tiles that were just joined, without having to relocate them in
        # the main thumbnail grid.
        main_win = find_main_window()
        if main_win and hasattr(main_win, "refresh_thumbnails"):
            main_win.refresh_thumbnails()
