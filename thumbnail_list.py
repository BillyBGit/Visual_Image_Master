"""
Thumbnail list widget for displaying image thumbnails.
"""

from PySide6.QtWidgets import (
    QListWidget,
    QMessageBox,
    QApplication,
    QAbstractItemView,
    QMenu,
    QInputDialog,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QFrame,
)
from PySide6.QtCore import Qt, QSize, QUrl, QMimeData, QItemSelectionModel
from PySide6.QtGui import QIcon, QDrag, QAction
from PIL import Image

from utils import find_main_window, reset_link_list, append_to_link_list
from preview_window import ImagePreviewWindow


class ManualCaptionDialog(QDialog):
    """Dialog for entering a caption to write alongside selected image files."""

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.setWindowTitle("Manual Caption")
        self.setMinimumWidth(480)
        self.setMinimumHeight(280)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        n = len(image_paths)
        layout.addWidget(
            QLabel(
                f"Caption text to write for {n} selected image{'s' if n != 1 else ''}:"
            )
        )

        self.text_edit = QTextEdit()
        self.text_edit.setAcceptRichText(False)
        self.text_edit.setPlaceholderText("Enter caption text here…")
        layout.addWidget(self.text_edit, stretch=1)

        # Mode group
        mode_box = QGroupBox("Write mode")
        mode_layout = QHBoxLayout(mode_box)
        self._mode_group = QButtonGroup(self)
        self._rb_overwrite = QRadioButton("Overwrite")
        self._rb_prepend = QRadioButton("Prepend")
        self._rb_append = QRadioButton("Append")
        self._rb_overwrite.setChecked(True)
        for rb in (self._rb_overwrite, self._rb_prepend, self._rb_append):
            self._mode_group.addButton(rb)
            mode_layout.addWidget(rb)
        mode_layout.addStretch()
        layout.addWidget(mode_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.text_edit.setFocus()

    @property
    def caption_text(self):
        return self.text_edit.toPlainText()

    @property
    def mode(self):
        if self._rb_prepend.isChecked():
            return "prepend"
        if self._rb_append.isChecked():
            return "append"
        return "overwrite"


class ImagePropertiesDialog(QDialog):
    """Dialog showing file/image properties for one or more selected thumbnails."""

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Properties")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        for i, path in enumerate(image_paths):
            if i > 0:
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.HLine)
                divider.setFrameShadow(QFrame.Shadow.Sunken)
                layout.addWidget(divider)
            layout.addLayout(self._build_properties_block(path))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _build_properties_block(self, path):
        block = QVBoxLayout()
        for label_text, value in self._get_image_info(path):
            row = QHBoxLayout()
            key_label = QLabel(f"{label_text}:")
            key_label.setStyleSheet("font-weight: bold;")
            key_label.setMinimumWidth(90)
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            value_label.setWordWrap(True)
            row.addWidget(key_label)
            row.addWidget(value_label, stretch=1)
            block.addLayout(row)
        return block

    @staticmethod
    def _get_image_info(path):
        """Gather path/format/dimensions/size/modified info for one file."""
        import os

        rows = [("Path", path)]
        try:
            with Image.open(path) as img:
                rows.append(("Format", img.format or "Unknown"))
                rows.append(("Dimensions", f"{img.width} x {img.height} px"))
                rows.append(("Color Mode", img.mode))
        except Exception:
            rows.append(("Format", "Unable to read"))
            rows.append(("Dimensions", "Unable to read"))

        try:
            size_bytes = os.path.getsize(path)
            rows.append(("File Size", ImagePropertiesDialog._format_size(size_bytes)))
        except OSError:
            rows.append(("File Size", "Unknown"))

        try:
            import datetime

            mtime = os.path.getmtime(path)
            modified = datetime.datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            rows.append(("Modified", modified))
        except OSError:
            pass

        return rows

    @staticmethod
    def _format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class ThumbnailList(QListWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumb_size = 256
        self.setIconSize(QSize(self.thumb_size, self.thumb_size))
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.itemDoubleClicked.connect(self.open_preview)

        # Tracks a pending Ctrl+click-on-item (see mousePressEvent) so a
        # following mouseMoveEvent/mouseReleaseEvent knows whether to
        # suppress Qt's native handling or hand off to it as a real drag.
        self._ctrl_click_item = None
        self._ctrl_click_start_pos = None

        # Store original stylesheet
        self._original_stylesheet = self.styleSheet()

        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def set_filter_active(self, active):
        """Set visual indication that filtering is active."""
        if active:
            # Blue border to indicate filter is active
            self.setStyleSheet("""
                QListWidget {
                    border: 3px solid #007FFF;
                    border-radius: 2px;
                }
            """)
        else:
            # Restore original stylesheet
            self.setStyleSheet(self._original_stylesheet)

    def mousePressEvent(self, event):
        """Handle Ctrl+click as an explicit single-item toggle.

        IconMode's default press handling can misinterpret a Ctrl+click with
        a pixel or two of mouse jitter between press and release as the
        start of a rubber-band drag-select, and IconMode's row-based
        hit-testing for that rubber band can end up matching the whole
        originating row instead of just the point under the cursor -
        spontaneously selecting everything to the left of the clicked
        thumbnail in that row, or (if several Ctrl+clicks are made in a
        row) an unpredictable-looking jumble across rows.

        super().mousePressEvent() is still called - skipping it entirely
        left Qt's own press/drag-start bookkeeping uninitialized, causing
        knock-on corruption on the *next* mouse interaction - so instead the
        exact toggle we want is simply re-applied immediately afterward,
        overriding whatever native selection decision Qt just made. See
        mouseMoveEvent for why a follow-up native move also has to be
        suppressed until it's unambiguously a real drag.
        """
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                was_selected = item.isSelected()
                super().mousePressEvent(event)
                item.setSelected(not was_selected)
                self.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
                self._ctrl_click_item = item
                self._ctrl_click_start_pos = event.position().toPoint()
                event.accept()
                return
        self._ctrl_click_item = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Suppress native move-driven selection churn following a
        Ctrl+click, until the mouse has moved far enough to count as an
        actual drag (at which point it's handed back to Qt so dragging the
        Ctrl-built selection back out of the grid still works)."""
        if self._ctrl_click_item is not None:
            moved = event.position().toPoint() - self._ctrl_click_start_pos
            if moved.manhattanLength() < QApplication.startDragDistance():
                event.accept()
                return
            self._ctrl_click_item = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Swallow the release that pairs with a suppressed Ctrl+click (no
        drag occurred) - everything it would do was already handled in
        mousePressEvent."""
        if self._ctrl_click_item is not None:
            self._ctrl_click_item = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Intercepts keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Delete:
            # Shift+Delete = permanent delete (no undo)
            permanent = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.delete_selected_files(permanent=permanent)
        elif (
            event.key() == Qt.Key.Key_R
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.rotate_selected_images()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Enter key opens preview, same as double-click
            current_item = self.currentItem()
            if current_item:
                self.open_preview(current_item)
        elif event.key() == Qt.Key.Key_Home:
            # Jump to first item
            if self.count() > 0:
                self.setCurrentRow(0)
                self.scrollToItem(self.item(0))
        elif event.key() == Qt.Key.Key_End:
            # Jump to last item
            if self.count() > 0:
                last = self.count() - 1
                self.setCurrentRow(last)
                self.scrollToItem(self.item(last))
        elif Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_5:
            self.rate_selected_images(event.key() - Qt.Key.Key_0)
        elif event.key() == Qt.Key.Key_0:
            self.rate_selected_images(None)
        else:
            super().keyPressEvent(event)

    def delete_selected_files(self, permanent=False):
        """Delete selected files.

        Args:
            permanent: If True, delete permanently (Shift+Delete). If False, allow undo.
        """
        import os

        selected_items = self.selectedItems()
        if not selected_items:
            return

        main_win = self.parent_app or find_main_window()
        should_confirm = True
        if main_win and hasattr(main_win, "conf_delete_act"):
            should_confirm = main_win.conf_delete_act.isChecked()

        do_delete = True
        if should_confirm:
            delete_type = "permanently delete" if permanent else "delete"
            confirm = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to {delete_type} {len(selected_items)} file(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            do_delete = confirm == QMessageBox.StandardButton.Yes

        if do_delete:
            # Remember the row of the first selected item to select next
            first_row = self.row(selected_items[0])

            # Start a new deletion batch for undo (clears previous undo state)
            if not permanent and main_win and hasattr(main_win, "delete_undo_manager"):
                main_win.delete_undo_manager.start_new_deletion()

            deleted_paths = []
            for item in selected_items:
                file_path = item.data(Qt.ItemDataRole.UserRole)
                try:
                    if os.path.exists(file_path):
                        if permanent:
                            # Permanent delete - no undo
                            os.remove(file_path)
                        else:
                            # Store for undo (moves to temp location)
                            if main_win and hasattr(main_win, "delete_undo_manager"):
                                main_win.delete_undo_manager.store_for_undo(file_path)
                            else:
                                os.remove(file_path)

                    deleted_paths.append(file_path)
                    self.takeItem(self.row(item))
                except Exception as e:
                    QMessageBox.warning(
                        self, "Error", f"Could not delete {file_path}:\n{e}"
                    )

            # Auto-select next image after deletion
            if deleted_paths and self.count() > 0:
                # Select the item at the same position, or the last item if we deleted the last items
                next_row = min(first_row, self.count() - 1)
                if next_row >= 0:
                    self.setCurrentRow(next_row)

            # Notify all open preview windows
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, ImagePreviewWindow):
                    widget.handle_external_deletion(deleted_paths)

            # Show status message
            if main_win and hasattr(main_win, "status_bar"):
                if permanent:
                    main_win.status_bar.showMessage(
                        f"Permanently deleted {len(deleted_paths)} file(s)", 3000
                    )
                else:
                    main_win.status_bar.showMessage(
                        f"Deleted {len(deleted_paths)} file(s) - Ctrl+Z to undo", 3000
                    )

    def rotate_selected_images(self):
        """Rotate selected images 90° clockwise and reload thumbnails."""
        import os

        selected_items = self.selectedItems()
        if not selected_items:
            return

        rotated_paths = []
        failed_count = 0

        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            try:
                # Load image
                img = Image.open(file_path)

                # Rotate 90° clockwise (which is -90° in PIL's coordinate system)
                rotated = img.rotate(-90, expand=True)

                # Save back to same file, preserving format and quality
                img_format = img.format
                if img_format in ["JPEG", "JPG"]:
                    rotated.save(
                        file_path, format=img_format, quality=95, optimize=True
                    )
                else:
                    rotated.save(file_path, format=img_format)

                rotated_paths.append(file_path)

            except Exception as e:
                failed_count += 1
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not rotate {os.path.basename(file_path)}:\n{e}",
                )

        # Reload thumbnails for rotated images
        if rotated_paths:
            # Create a fresh QIcon for each rotated image
            # We need to bypass Qt's icon cache by creating a new QPixmap
            from PySide6.QtGui import QPixmap

            for item in selected_items:
                file_path = item.data(Qt.ItemDataRole.UserRole)
                if file_path in rotated_paths:
                    # Load fresh pixmap and create new icon to bypass cache
                    pixmap = QPixmap(file_path)
                    item.setIcon(QIcon(pixmap))

            # Force a complete visual update
            self.scheduleDelayedItemsLayout()
            self.viewport().update()
            QApplication.processEvents()

            # Show status message
            main_win = self.parent_app or find_main_window()
            if main_win:
                success_count = len(rotated_paths)
                main_win.status_bar.showMessage(
                    f"Rotated {success_count} image(s) 90° clockwise", 3000
                )

    def resize_to_padded_square(self, size=1024):
        """Resize selected images to size×size with black padding, retaining aspect ratio."""
        import os
        from PySide6.QtGui import QPixmap

        selected_items = self.selectedItems()
        if not selected_items:
            return

        count = len(selected_items)
        confirm = QMessageBox.question(
            self,
            "Resize Images",
            f"Resize {count} image(s) to {size}×{size} with black padding?\n\nThis overwrites the original files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        main_win = self.parent_app or find_main_window()
        success_count = 0

        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            try:
                from io import BytesIO

                # Read entire file into memory first so there is no open file
                # handle when we write back to the same path
                with open(file_path, "rb") as fh:
                    raw = fh.read()

                img = Image.open(BytesIO(raw))
                img.load()

                # Always work in RGB — handles L, P, RGBA, LA, CMYK, etc.
                if img.mode != "RGB":
                    img = img.convert("RGB")

                w, h = img.size
                if w == 0 or h == 0:
                    raise ValueError(f"Image has zero dimension ({w}×{h})")

                # Scale to fit within size×size keeping aspect ratio (up or down)
                scale = min(size / w, size / h)
                new_w = min(size, max(1, round(w * scale)))
                new_h = min(size, max(1, round(h * scale)))
                img = img.resize((new_w, new_h), Image.LANCZOS)

                # Paste centred onto black canvas
                canvas = Image.new("RGB", (size, size), (0, 0, 0))
                canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))

                # Determine output format from file extension (more reliable
                # than img.format which becomes None after PIL operations)
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".png":
                    canvas.save(file_path, format="PNG")
                elif ext == ".webp":
                    canvas.save(file_path, format="WEBP", quality=95)
                elif ext in (".bmp",):
                    canvas.save(file_path, format="BMP")
                elif ext in (".tiff", ".tif"):
                    canvas.save(file_path, format="TIFF")
                else:
                    canvas.save(file_path, format="JPEG", quality=95)

                item.setIcon(QIcon(QPixmap(file_path)))
                success_count += 1

            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not resize {os.path.basename(file_path)}:\n{e}",
                )

        if success_count > 0:
            self.scheduleDelayedItemsLayout()
            self.viewport().update()
            if main_win and hasattr(main_win, "status_bar"):
                main_win.status_bar.showMessage(
                    f"Resized {success_count} image(s) to {size}×{size}", 3000
                )

    def open_preview(self, item):
        all_paths = [
            self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())
        ]
        current_idx = self.row(item)
        preview = ImagePreviewWindow(all_paths, current_idx, self.parent_app)
        preview.show()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            self.thumb_size = max(64, min(512, self.thumb_size + (delta // 8)))
            new_qsize = QSize(self.thumb_size, self.thumb_size)
            self.setIconSize(new_qsize)
            self.setGridSize(QSize(self.thumb_size + 20, self.thumb_size + 40))
            for i in range(self.count()):
                item = self.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                item.setIcon(QIcon(path))
        else:
            super().wheelEvent(event)

    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        urls = [
            QUrl.fromLocalFile(i.data(Qt.ItemDataRole.UserRole))
            for i in items
            if i.data(Qt.ItemDataRole.UserRole)
        ]
        mime_data.setUrls(urls)
        drag.setMimeData(mime_data)

        # Drag pixmap from the first selected thumbnail — required for external
        # apps (file managers, image editors) to recognise and accept the drag.
        icon = items[0].icon()
        if not icon.isNull():
            px = icon.pixmap(96, 96)
            drag.setPixmap(px)
            drag.setHotSpot(px.rect().center())

        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

    def show_context_menu(self, position):
        """Show context menu for thumbnail operations."""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        menu = QMenu(self)

        # Move to selected
        move_selected_act = QAction("Move to Selected", self)
        move_selected_act.triggered.connect(lambda: self.move_to_selected())
        menu.addAction(move_selected_act)

        # Move to new
        move_new_act = QAction("Move to New...", self)
        move_new_act.triggered.connect(lambda: self.move_to_new())
        menu.addAction(move_new_act)

        menu.addSeparator()

        # Copy to selected
        copy_selected_act = QAction("Copy to Selected", self)
        copy_selected_act.triggered.connect(lambda: self.copy_to_selected())
        menu.addAction(copy_selected_act)

        # Copy to new
        copy_new_act = QAction("Copy to New...", self)
        copy_new_act.triggered.connect(lambda: self.copy_to_new())
        menu.addAction(copy_new_act)

        menu.addSeparator()

        # Copy Path
        copy_path_act = QAction("Copy Path", self)
        copy_path_act.triggered.connect(self.copy_selected_paths)
        menu.addAction(copy_path_act)

        # Link list (clipboard-based running list of paths)
        start_link_act = QAction("Start Link List", self)
        start_link_act.triggered.connect(self.start_link_list)
        menu.addAction(start_link_act)

        add_link_act = QAction("Add to Link List", self)
        add_link_act.triggered.connect(self.add_to_link_list)
        menu.addAction(add_link_act)

        menu.addSeparator()

        # Rating
        rate_menu = QMenu("Rate", self)
        for score in range(1, 6):
            stars = "★" * score + "☆" * (5 - score)
            rate_act = QAction(f"{stars}  ({score})", self)
            rate_act.triggered.connect(
                lambda checked=False, s=score: self.rate_selected_images(s)
            )
            rate_menu.addAction(rate_act)
        rate_menu.addSeparator()
        clear_rating_act = QAction("Clear Rating", self)
        clear_rating_act.triggered.connect(lambda: self.rate_selected_images(None))
        rate_menu.addAction(clear_rating_act)
        menu.addMenu(rate_menu)

        menu.addSeparator()

        # Auto Caption
        caption_act = QAction("Auto Caption...", self)
        caption_act.triggered.connect(self.show_autocaption_dialog)
        menu.addAction(caption_act)

        # Manual Caption
        manual_caption_act = QAction("Manual Caption...", self)
        manual_caption_act.triggered.connect(self.show_manual_caption_dialog)
        menu.addAction(manual_caption_act)

        # Enhance
        enhance_act = QAction("Enhance...", self)
        enhance_act.triggered.connect(self.show_enhance_dialog)
        menu.addAction(enhance_act)

        # Join Images
        join_act = QAction("Join Images...", self)
        join_act.triggered.connect(self.show_join_images_dialog)
        join_act.setEnabled(2 <= len(selected_items) <= 6)
        if not join_act.isEnabled():
            join_act.setToolTip("Select between 2 and 6 images to join.")
        menu.addAction(join_act)

        # Resize to padded square
        resize_act = QAction("Resize to 1024×1024 (padded)", self)
        resize_act.triggered.connect(lambda: self.resize_to_padded_square())
        menu.addAction(resize_act)

        menu.addSeparator()

        # Sort by similarity to this image (uses the first selected thumbnail
        # as the reference face)
        similarity_sort_act = QAction("Sort by Similarity to This Image", self)
        similarity_sort_act.triggered.connect(self.sort_by_similarity_to_selected)
        menu.addAction(similarity_sort_act)

        # Stronger, multi-reference version: use every selected thumbnail as
        # the reference gallery instead of just one
        similarity_multi_act = QAction("Sort by Similarity to Selected Images", self)
        similarity_multi_act.triggered.connect(self.sort_by_similarity_to_selection)
        similarity_multi_act.setEnabled(len(selected_items) >= 2)
        if not similarity_multi_act.isEnabled():
            similarity_multi_act.setToolTip(
                "Select 2 or more images to use as the reference set."
            )
        menu.addAction(similarity_multi_act)

        # Discriminative version: selected thumbnails as the positive
        # reference gallery, plus a prompted folder of negative images
        # (other people/subjects to actively rank down)
        similarity_exclude_act = QAction(
            "Sort by Similarity to Selected Images (+ Negative Folder)...", self
        )
        similarity_exclude_act.triggered.connect(
            self.sort_by_similarity_to_selection_with_negatives
        )
        similarity_exclude_act.setEnabled(len(selected_items) >= 2)
        if not similarity_exclude_act.isEnabled():
            similarity_exclude_act.setToolTip(
                "Select 2 or more images to use as the reference set."
            )
        menu.addAction(similarity_exclude_act)

        menu.addSeparator()

        # Delete
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.delete_selected_files)
        menu.addAction(delete_act)

        menu.addSeparator()

        # Properties
        properties_act = QAction("Properties", self)
        properties_act.triggered.connect(self.show_properties_dialog)
        menu.addAction(properties_act)

        # Show menu at cursor position
        menu.exec(self.mapToGlobal(position))

    def copy_selected_paths(self):
        """Copy the full file path(s) of the selected thumbnail(s) to the clipboard."""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        QApplication.clipboard().setText("\n".join(paths))

    def start_link_list(self):
        """Start a fresh link list seeded with the selected thumbnail path(s)."""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        reset_link_list(self, "\n".join(paths))

    def add_to_link_list(self):
        """Append the selected thumbnail path(s) to the link list."""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        append_to_link_list("\n".join(paths))

    def rate_selected_images(self, rating):
        """Set (or clear, if rating is None) the rating for selected thumbnails."""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        from rating_manager import get_rating_manager

        mgr = get_rating_manager()
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                mgr.set_rating(path, rating)

        self._refresh_rating_display(selected_items)

        main_win = self.parent_app or find_main_window()
        if main_win and hasattr(main_win, "status_bar"):
            if rating:
                main_win.status_bar.showMessage(
                    f"Rated {len(selected_items)} image(s) {rating}/5", 3000
                )
            else:
                main_win.status_bar.showMessage(
                    f"Cleared rating for {len(selected_items)} image(s)", 3000
                )

    def _refresh_rating_display(self, items):
        """Update items' displayed text to reflect their current rating."""
        from rating_manager import get_rating_manager, combine_display_text

        mgr = get_rating_manager()
        for item in items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if not path:
                continue
            base_text = item.data(Qt.ItemDataRole.UserRole + 1) or ""
            rating = mgr.get_rating(path)
            item.setText(combine_display_text(base_text, rating))

    def refresh_rating_display_for_path(self, path):
        """Update the thumbnail's displayed star suffix for a single path,
        if it's currently shown. Called by the preview window after rating
        an image there, so the thumbnail grid doesn't go stale until the
        next full refresh_thumbnails()."""
        if not path:
            return
        import os

        target = os.path.normpath(os.path.abspath(path))
        for i in range(self.count()):
            item = self.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path and os.path.normpath(os.path.abspath(item_path)) == target:
                self._refresh_rating_display([item])
                return

    def show_properties_dialog(self):
        """Show a Properties popup with path/format/dimensions for the selection."""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        dialog = ImagePropertiesDialog(paths, self)
        dialog.exec()

    def get_selected_dest_folder(self):
        """Get the currently selected destination folder from main window."""
        main_win = self.parent_app or find_main_window()
        if not main_win:
            return None

        current_index = main_win.dest_tree.currentIndex()
        if current_index.isValid():
            return main_win.dest_model.filePath(
                main_win.dest_proxy.mapToSource(current_index)
            )
        return main_win.dest_root_path

    def move_to_selected(self):
        """Move selected files to the currently selected destination folder."""
        dest_folder = self.get_selected_dest_folder()
        if not dest_folder:
            QMessageBox.warning(
                self, "No Destination", "Please select a destination folder first."
            )
            return

        self.transfer_files(dest_folder, is_copy=False)

    def copy_to_selected(self):
        """Copy selected files to the currently selected destination folder."""
        dest_folder = self.get_selected_dest_folder()
        if not dest_folder:
            QMessageBox.warning(
                self, "No Destination", "Please select a destination folder first."
            )
            return

        self.transfer_files(dest_folder, is_copy=True)

    def move_to_new(self):
        """Move selected files to a new/specified folder."""
        self.transfer_to_new(is_copy=False)

    def copy_to_new(self):
        """Copy selected files to a new/specified folder."""
        self.transfer_to_new(is_copy=True)

    def transfer_to_new(self, is_copy):
        """Prompt for new folder path and transfer files."""
        import os

        # Get current destination folder for relative path resolution
        current_dest = self.get_selected_dest_folder()

        # Prompt user for path
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Enter Destination Path")
        dialog.setLabelText(
            "Enter path (absolute from / or relative to selected destination):"
        )
        dialog.setTextValue("")
        dialog.resize(500, 150)  # Set reasonable size

        ok = dialog.exec()
        path = dialog.textValue()

        if not ok or not path:
            return

        path = path.strip()

        # Resolve path
        if os.path.isabs(path):
            # Absolute path
            dest_folder = path
        else:
            # Relative path - resolve relative to current destination
            if not current_dest:
                QMessageBox.warning(
                    self,
                    "No Base Folder",
                    "Cannot use relative path without a selected destination folder.",
                )
                return
            dest_folder = os.path.join(current_dest, path)

        # Create directory if it doesn't exist
        if not os.path.exists(dest_folder):
            try:
                os.makedirs(dest_folder)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not create directory:\n{e}")
                return

        if not os.path.isdir(dest_folder):
            QMessageBox.warning(
                self, "Invalid Path", f"Path is not a directory:\n{dest_folder}"
            )
            return

        self.transfer_files(dest_folder, is_copy)

    def transfer_files(self, dest_folder, is_copy):
        """Transfer selected files to destination folder."""
        import os

        selected_items = self.selectedItems()
        if not selected_items:
            return

        main_win = self.parent_app or find_main_window()
        if not main_win:
            return

        # Start a new operation batch for undo tracking
        operation_type = "copy" if is_copy else "move"
        main_win.start_operation_batch(operation_type)

        transferred_paths = []
        failed_count = 0

        for item in selected_items:
            src_path = item.data(Qt.ItemDataRole.UserRole)
            try:
                main_win.smart_transfer(src_path, dest_folder, is_copy)
                if not is_copy:
                    transferred_paths.append(src_path)
            except Exception as e:
                failed_count += 1
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not transfer {os.path.basename(src_path)}:\n{e}",
                )

        # Remove transferred items from thumbnail list (only for moves)
        if not is_copy and transferred_paths:
            main_win.remove_thumbnails_by_path(transferred_paths)

        # Show status message
        operation = "Copied" if is_copy else "Moved"
        success_count = len(selected_items) - failed_count
        if success_count > 0:
            main_win.status_bar.showMessage(
                f"{operation} {success_count} file(s) to {os.path.basename(dest_folder)}",
                3000,
            )

    def show_autocaption_dialog(self):
        """Show the auto-caption dialog for selected images."""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        # Get list of image paths
        image_paths = []
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_paths.append(path)

        if not image_paths:
            QMessageBox.warning(self, "No Images", "No valid images selected.")
            return

        # Show the autocaption dialog
        from autocaption_dialog import AutoCaptionDialog

        dialog = AutoCaptionDialog(image_paths, self)
        dialog.exec()

    def show_manual_caption_dialog(self):
        """Show the manual caption dialog for selected images."""
        import os

        selected_items = self.selectedItems()
        if not selected_items:
            return

        image_paths = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in selected_items
            if item.data(Qt.ItemDataRole.UserRole)
        ]
        if not image_paths:
            return

        dialog = ManualCaptionDialog(image_paths, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        text = dialog.caption_text
        mode = dialog.mode
        if not text and mode == "overwrite":
            # Overwriting with empty string is allowed (clears the caption)
            pass

        written, skipped, errors = 0, 0, []
        for img_path in image_paths:
            base = os.path.splitext(img_path)[0]
            txt_path = base + ".txt"
            try:
                existing = ""
                if os.path.exists(txt_path) and mode != "overwrite":
                    with open(txt_path, "r", encoding="utf-8") as fh:
                        existing = fh.read()

                if mode == "prepend":
                    content = text + (" " if existing and text else "") + existing
                elif mode == "append":
                    content = existing + (" " if existing and text else "") + text
                else:
                    content = text

                with open(txt_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                written += 1
            except Exception as e:
                errors.append(f"{os.path.basename(img_path)}: {e}")
                skipped += 1

        if errors:
            QMessageBox.warning(
                self,
                "Caption Errors",
                f"Wrote {written} caption(s). {skipped} failed:\n" + "\n".join(errors),
            )
        else:
            main_win = self.parent_app or find_main_window()
            if main_win and hasattr(main_win, "status_bar"):
                main_win.status_bar.showMessage(
                    f"Caption written to {written} file(s)  [{mode}]", 3000
                )

    def show_enhance_dialog(self):
        """Show the enhancement dialog for selected images."""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        # Get list of image paths
        image_paths = []
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_paths.append(path)

        if not image_paths:
            QMessageBox.warning(self, "No Images", "No valid images selected.")
            return

        # Show the enhance dialog
        from enhance_dialog import EnhanceDialog

        dialog = EnhanceDialog(image_paths, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh thumbnails if images were modified
            main_win = self.parent_app or find_main_window()
            if main_win and hasattr(main_win, "refresh_thumbnails"):
                main_win.refresh_thumbnails()

    def sort_by_similarity_to_selected(self):
        """Sort the thumbnail grid by facial similarity to the first
        selected image, closest match first."""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        main_win = self.parent_app or find_main_window()
        if main_win and hasattr(main_win, "set_similarity_sort"):
            main_win.set_similarity_sort(path)

    def sort_by_similarity_to_selection(self):
        """Sort the thumbnail grid by nearest-neighbor facial similarity to
        ALL currently selected images - a stronger, multi-reference version
        of Sort by Similarity to This Image."""
        selected_items = self.selectedItems()
        if len(selected_items) < 2:
            return
        paths = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in selected_items
            if item.data(Qt.ItemDataRole.UserRole)
        ]
        if len(paths) < 2:
            return

        main_win = self.parent_app or find_main_window()
        if main_win and hasattr(main_win, "set_similarity_sort_multi"):
            main_win.set_similarity_sort_multi(paths)

    def sort_by_similarity_to_selection_with_negatives(self):
        """Like sort_by_similarity_to_selection, but also prompts for a
        folder of negative images - other people/subjects to actively rank
        down, not just not-matched. See
        ImageSorter.set_similarity_sort_discriminative."""
        selected_items = self.selectedItems()
        if len(selected_items) < 2:
            return
        paths = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in selected_items
            if item.data(Qt.ItemDataRole.UserRole)
        ]
        if len(paths) < 2:
            return

        main_win = self.parent_app or find_main_window()
        if not main_win or not hasattr(main_win, "set_similarity_sort_discriminative"):
            return

        negative_paths = main_win._prompt_for_image_folder(
            "Choose Negative Image Folder (other people/images to rank down)"
        )
        if negative_paths is None:
            return

        main_win.set_similarity_sort_discriminative(paths, negative_paths)

    def show_join_images_dialog(self):
        """Show the join-images dialog for 2-6 selected images."""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        image_paths = []
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_paths.append(path)

        if not (2 <= len(image_paths) <= 6):
            QMessageBox.warning(
                self, "Join Images", "Select between 2 and 6 images to join."
            )
            return

        from join_images_dialog import JoinImagesDialog

        dialog = JoinImagesDialog(image_paths, self)
        # The dialog stays open after a join/move/delete (so the user can
        # keep managing the source tiles) and refreshes the main grid live
        # as those happen; refresh once more here in case it closes early.
        dialog.exec()
        main_win = self.parent_app or find_main_window()
        if main_win and hasattr(main_win, "refresh_thumbnails"):
            main_win.refresh_thumbnails()
