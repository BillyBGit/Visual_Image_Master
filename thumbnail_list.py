"""
Thumbnail list widget for displaying image thumbnails.
"""

from PySide6.QtWidgets import (QListWidget, QListWidgetItem, QMessageBox, QApplication,
                               QAbstractItemView, QMenu, QInputDialog)
from PySide6.QtCore import Qt, QSize, QUrl, QMimeData, QPoint
from PySide6.QtGui import QIcon, QDrag, QPalette, QColor, QAction
from PIL import Image

from utils import find_main_window
from preview_window import ImagePreviewWindow


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

    def keyPressEvent(self, event):
        """Intercepts keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Delete:
            # Shift+Delete = permanent delete (no undo)
            permanent = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.delete_selected_files(permanent=permanent)
        elif event.key() == Qt.Key.Key_R and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.rotate_selected_images()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Enter key opens preview, same as double-click
            current_item = self.currentItem()
            if current_item:
                self.open_preview(current_item)
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
        if main_win and hasattr(main_win, 'conf_delete_act'):
            should_confirm = main_win.conf_delete_act.isChecked()

        do_delete = True
        if should_confirm:
            delete_type = "permanently delete" if permanent else "delete"
            confirm = QMessageBox.question(
                self, "Confirm Delete",
                f"Are you sure you want to {delete_type} {len(selected_items)} file(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            do_delete = (confirm == QMessageBox.StandardButton.Yes)

        if do_delete:
            # Remember the row of the first selected item to select next
            first_row = self.row(selected_items[0])

            # Start a new deletion batch for undo (clears previous undo state)
            if not permanent and main_win and hasattr(main_win, 'delete_undo_manager'):
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
                            if main_win and hasattr(main_win, 'delete_undo_manager'):
                                main_win.delete_undo_manager.store_for_undo(file_path)
                            else:
                                os.remove(file_path)

                    deleted_paths.append(file_path)
                    self.takeItem(self.row(item))
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete {file_path}:\n{e}")

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
            if main_win and hasattr(main_win, 'status_bar'):
                if permanent:
                    main_win.status_bar.showMessage(f"Permanently deleted {len(deleted_paths)} file(s)", 3000)
                else:
                    main_win.status_bar.showMessage(f"Deleted {len(deleted_paths)} file(s) - Ctrl+Z to undo", 3000)

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
                if img_format in ['JPEG', 'JPG']:
                    rotated.save(file_path, format=img_format, quality=95, optimize=True)
                else:
                    rotated.save(file_path, format=img_format)

                rotated_paths.append(file_path)

            except Exception as e:
                failed_count += 1
                QMessageBox.warning(self, "Error", f"Could not rotate {os.path.basename(file_path)}:\n{e}")

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
                    f"Rotated {success_count} image(s) 90° clockwise",
                    3000
                )

    def open_preview(self, item):
        all_paths = [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]
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
        drag = QDrag(self)
        mime_data = QMimeData()
        urls = [QUrl.fromLocalFile(i.data(Qt.ItemDataRole.UserRole)) for i in self.selectedItems()]
        mime_data.setUrls(urls)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction | Qt.DropAction.CopyAction)

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

        # Delete
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.delete_selected_files)
        menu.addAction(delete_act)

        # Show menu at cursor position
        menu.exec(self.mapToGlobal(position))

    def get_selected_dest_folder(self):
        """Get the currently selected destination folder from main window."""
        main_win = self.parent_app or find_main_window()
        if not main_win:
            return None

        current_index = main_win.dest_tree.currentIndex()
        if current_index.isValid():
            return main_win.dest_model.filePath(current_index)
        return main_win.dest_root_path

    def move_to_selected(self):
        """Move selected files to the currently selected destination folder."""
        dest_folder = self.get_selected_dest_folder()
        if not dest_folder:
            QMessageBox.warning(self, "No Destination", "Please select a destination folder first.")
            return

        self.transfer_files(dest_folder, is_copy=False)

    def copy_to_selected(self):
        """Copy selected files to the currently selected destination folder."""
        dest_folder = self.get_selected_dest_folder()
        if not dest_folder:
            QMessageBox.warning(self, "No Destination", "Please select a destination folder first.")
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
        dialog.setLabelText("Enter path (absolute from / or relative to selected destination):")
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
                QMessageBox.warning(self, "No Base Folder",
                    "Cannot use relative path without a selected destination folder.")
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
            QMessageBox.warning(self, "Invalid Path", f"Path is not a directory:\n{dest_folder}")
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
        operation_type = 'copy' if is_copy else 'move'
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
                QMessageBox.warning(self, "Error", f"Could not transfer {os.path.basename(src_path)}:\n{e}")

        # Remove transferred items from thumbnail list (only for moves)
        if not is_copy and transferred_paths:
            main_win.remove_thumbnails_by_path(transferred_paths)

        # Show status message
        operation = "Copied" if is_copy else "Moved"
        success_count = len(selected_items) - failed_count
        if success_count > 0:
            main_win.status_bar.showMessage(
                f"{operation} {success_count} file(s) to {os.path.basename(dest_folder)}",
                3000
            )
