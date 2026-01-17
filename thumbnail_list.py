"""
Thumbnail list widget for displaying image thumbnails.
"""

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QMessageBox, QApplication, QAbstractItemView
from PySide6.QtCore import Qt, QSize, QUrl, QMimeData
from PySide6.QtGui import QIcon, QDrag, QPalette, QColor

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

    def set_filter_active(self, active):
        """Set visual indication that filtering is active."""
        if active:
            # Azure blue outline when filter is active
            self.setStyleSheet("""
                QListWidget {
                    border: 3px solid #007FFF;
                }
            """)
        else:
            # Restore original stylesheet
            self.setStyleSheet(self._original_stylesheet)

    def keyPressEvent(self, event):
        """Intercepts the Delete key to trigger file deletion."""
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_files()
        else:
            super().keyPressEvent(event)

    def delete_selected_files(self):
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
            confirm = QMessageBox.question(
                self, "Confirm Delete",
                f"Are you sure you want to permanently delete {len(selected_items)} file(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            do_delete = (confirm == QMessageBox.StandardButton.Yes)

        if do_delete:
            deleted_paths = []
            for item in selected_items:
                file_path = item.data(Qt.ItemDataRole.UserRole)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    deleted_paths.append(file_path)
                    self.takeItem(self.row(item))
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete {file_path}:\n{e}")

            # Notify all open preview windows
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, ImagePreviewWindow):
                    widget.handle_external_deletion(deleted_paths)

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
