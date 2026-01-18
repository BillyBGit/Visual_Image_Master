"""
Folder tree widget and delegate for displaying directory structures with image counts.
"""

import os
import shutil
from PySide6.QtWidgets import (QTreeView, QStyledItemDelegate, QMenu, QInputDialog,
                               QLineEdit, QMessageBox, QAbstractItemView, QApplication,
                               QFileSystemModel, QHeaderView)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QFont
from PySide6.QtGui import QDropEvent


class FolderCountDelegate(QStyledItemDelegate):
    """Custom delegate to display folder names with image counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.show_counts = False
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif', '.tiff', '.bmp')
        self._count_cache = {}

    def set_show_counts(self, show):
        """Toggle count display and clear cache."""
        self.show_counts = show
        self._count_cache.clear()

    def get_image_count(self, path):
        """Count image files in a directory with caching."""
        if not self.show_counts:
            return 0

        if path in self._count_cache:
            return self._count_cache[path]

        try:
            if os.path.isdir(path):
                count = sum(1 for f in os.listdir(path)
                           if f.lower().endswith(self.valid_extensions))
                self._count_cache[path] = count
                return count
        except (PermissionError, OSError):
            pass

        return 0

    def displayText(self, value, locale):
        """Override display text to append image count."""
        if not self.show_counts:
            return super().displayText(value, locale)

        # Get the tree view
        tree_view = self.parent()
        if not tree_view or not isinstance(tree_view, QTreeView):
            return value

        model = tree_view.model()
        if not model or not isinstance(model, QFileSystemModel):
            return value

        # Search for the index with this display value
        root_index = tree_view.rootIndex()
        for row in range(model.rowCount(root_index)):
            index = model.index(row, 0, root_index)
            if model.data(index, Qt.ItemDataRole.DisplayRole) == value:
                path = model.filePath(index)
                count = self.get_image_count(path)
                if count > 0:
                    return f"{value} ({count})"
                break

        return value


class FolderTree(QTreeView):
    def __init__(self, parent_app, is_source=False):
        super().__init__()
        self.parent_app = parent_app
        self.is_source = is_source
        self.header().hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.expanded.connect(lambda: self.resizeColumnToContents(0))

        # Set up custom delegate for showing counts
        self.count_delegate = FolderCountDelegate(self)
        self.setItemDelegate(self.count_delegate)

        # Zoom level (font size multiplier)
        self.zoom_level = 1.0  # 1.0 = default, range 0.8 to 2.0
        self.apply_zoom()

    def set_show_counts(self, show):
        """Update the delegate to show/hide counts and refresh display."""
        self.count_delegate.set_show_counts(show)
        self.viewport().update()

    def apply_zoom(self):
        """Apply current zoom level to the tree view."""
        # Get default font
        font = self.font()

        # Calculate new size based on zoom level
        base_size = 10  # Default font size
        new_size = int(base_size * self.zoom_level)
        font.setPointSize(new_size)
        self.setFont(font)

        # Adjust icon size proportionally
        base_icon_size = 16  # Default icon size
        new_icon_size = int(base_icon_size * self.zoom_level)
        self.setIconSize(QSize(new_icon_size, new_icon_size))

        # Adjust row height
        if self.header():
            base_height = 20
            new_height = int(base_height * self.zoom_level)
            self.header().setDefaultSectionSize(new_height)

        # Refresh the view
        self.resizeColumnToContents(0)
        self.viewport().update()

    def wheelEvent(self, event):
        """Handle Ctrl+wheel for zooming."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()

            # Adjust zoom level
            if delta > 0:
                self.zoom_level = min(2.0, self.zoom_level + 0.1)
            else:
                self.zoom_level = max(0.8, self.zoom_level - 0.1)

            # Apply the new zoom
            self.apply_zoom()

            # Save settings if available
            if self.parent_app and hasattr(self.parent_app, '_settings_loaded'):
                if self.parent_app._settings_loaded:
                    self.parent_app.write_settings()

            event.accept()
        else:
            # Normal scrolling
            super().wheelEvent(event)

    def set_model_and_clean_columns(self, model):
        self.setModel(model)
        # Connect selection signal AFTER the model is set
        if self.is_source:
            self.selectionModel().selectionChanged.connect(self.on_selection_changed)
        for i in range(1, 4):
            self.hideColumn(i)
        self.resizeColumnToContents(0)

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if indexes:
            # Filter for column 0 to avoid redundant triggers
            idx = next((i for i in indexes if i.column() == 0), None)
            if idx:
                # Reference the parent_app for these methods
                if self.parent_app and hasattr(self.parent_app, 'on_source_folder_changed'):
                    self.parent_app.on_source_folder_changed(idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_folder_prompt()
        else:
            super().keyPressEvent(event)

    def dragMoveEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if index.isValid():
            self.setCurrentIndex(index)
            QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        QApplication.restoreOverrideCursor()
        index = self.indexAt(event.position().toPoint())
        if not index.isValid():
            return

        dest_path = self.model().filePath(index)
        urls = event.mimeData().urls()
        is_copy = event.modifiers() & Qt.KeyboardModifier.ControlModifier

        # Start a new operation batch for undo tracking
        if self.parent_app:
            operation_type = 'copy' if is_copy else 'move'
            self.parent_app.start_operation_batch(operation_type)

        moved_files = []
        for url in urls:
            source_path = url.toLocalFile()
            if self.parent_app:
                self.parent_app.smart_transfer(source_path, dest_path, bool(is_copy))
            # Only remove from UI if it was a MOVE operation (not a copy)
            if not is_copy:
                moved_files.append(source_path)

        event.acceptProposedAction()

        # Only remove items if files were actually moved away
        if moved_files and self.parent_app:
            self.parent_app.remove_thumbnails_by_path(moved_files)

        # Refresh counts after file operations
        if self.count_delegate.show_counts:
            self.count_delegate._count_cache.clear()
            self.viewport().update()

    def show_context_menu(self, point):
        index = self.indexAt(point)
        menu = QMenu(self)
        new_folder_act = QAction("New Folder", self)
        new_folder_act.triggered.connect(lambda: self.create_folder_prompt(index))
        menu.addAction(new_folder_act)
        if index.isValid():
            rename_act = QAction("Rename", self)
            rename_act.triggered.connect(lambda: self.rename_prompt(index))
            menu.addAction(rename_act)
            delete_act = QAction("Delete Selection", self)
            delete_act.triggered.connect(self.delete_folder_prompt)
            menu.addAction(delete_act)
        menu.exec(self.viewport().mapToGlobal(point))

    def create_folder_prompt(self, index):
        parent_p = self.model().filePath(index) if index.isValid() else self.model().rootPath()
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:", QLineEdit.EchoMode.Normal, "New Folder")
        if ok and name:
            new_p = os.path.join(parent_p, name)
            if not os.path.exists(new_p):
                os.makedirs(new_p)

    def rename_prompt(self, index):
        old_path = self.model().filePath(index)
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", QLineEdit.EchoMode.Normal, os.path.basename(old_path))
        if ok and new_name:
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not rename: {e}")

    def delete_folder_prompt(self):
        selected_indexes = [i for i in self.selectedIndexes() if i.column() == 0]
        if not selected_indexes:
            return
        paths = sorted([self.model().filePath(i) for i in selected_indexes], reverse=True)
        reply = QMessageBox.question(
            self, 'Delete',
            f"Delete {len(paths)} folder(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for path in paths:
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete {path}: {e}")
