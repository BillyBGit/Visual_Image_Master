"""
Folder tree widget and delegate for displaying directory structures with image counts.
"""

import os
import re
import shutil
from PySide6.QtWidgets import (
    QTreeView,
    QStyledItemDelegate,
    QMenu,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QAbstractItemView,
    QApplication,
    QFileSystemModel,
)
from PySide6.QtCore import Qt, QSize, QSortFilterProxyModel, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtGui import QDropEvent


class FolderSortFilterProxyModel(QSortFilterProxyModel):
    """Proxy model that adds name-filter and name/date-sort to a QFileSystemModel."""

    # Emitted once a filter change (including any prefetch below) has been
    # fully applied. invalidateFilter() can leave a previously-captured root
    # QModelIndex stale, silently resetting the QTreeView to show the whole
    # filesystem from "/" - the owner (ImageSorter) listens for this to
    # re-assert the tree's scoped root index from the known path string.
    filter_applied = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_text = ""
        self._regex_mode = False
        self._scope_root = None  # folder the tree is scoped to (plain path string)
        self._sort_mode = "name"  # "name" or "date_modified"
        self._sort_order = Qt.SortOrder.AscendingOrder
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setDynamicSortFilter(True)
        # Without this, a folder whose own name doesn't match the filter is
        # hidden along with everything under it - even if a descendant does
        # match. Recursive filtering keeps the ancestor path to any match
        # visible instead of blanking the whole tree.
        self.setRecursiveFilteringEnabled(True)

        # Debounced prefetch: QFileSystemModel only knows about directories
        # it has already been asked about, so recursive filtering can only
        # reveal matches among rows it's already fetched. Before applying a
        # filter we walk the scope root ourselves and touch every matching
        # path via model.index(), forcing it (and its ancestors) to load.
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.timeout.connect(self._prefetch_and_apply)

    def setSourceModel(self, model):
        super().setSourceModel(model)
        # Safe to sort now that the source model is attached
        self.sort(0, self._sort_order)

    def filePath(self, proxy_index):
        if not proxy_index.isValid():
            return ""
        return self.sourceModel().filePath(self.mapToSource(proxy_index))

    def set_scope_root(self, path):
        """Record the folder the tree is scoped to (as a plain path string).

        Used both as the starting point for the prefetch walk and by the
        owner to re-derive a fresh root QModelIndex after filtering.
        """
        self._scope_root = path

    def set_filter_text(self, text):
        self._filter_text = text.strip()
        self._schedule_apply()

    def set_regex_mode(self, enabled):
        self._regex_mode = enabled
        self._schedule_apply()

    def _schedule_apply(self):
        if not self._filter_text:
            self.invalidateFilter()
            self.filter_applied.emit()
            return
        self._prefetch_timer.start(250)

    def _prefetch_and_apply(self):
        self._prefetch_matches()
        self.invalidateFilter()
        self.filter_applied.emit()

    def _prefetch_matches(self):
        model = self.sourceModel()
        if model is None or not self._scope_root or not os.path.isdir(self._scope_root):
            return
        self._walk_and_populate(model, self._scope_root)

    def _walk_and_populate(self, model, directory):
        """Recurse into directory, forcing any name-matching subdirectory
        (and thus its ancestors) into the source model's fetched cache."""
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if self._name_matches(entry.name):
                model.index(entry.path)
            self._walk_and_populate(model, entry.path)

    def set_sort_mode(self, mode, ascending=True):
        self._sort_mode = mode
        self._sort_order = (
            Qt.SortOrder.AscendingOrder if ascending else Qt.SortOrder.DescendingOrder
        )
        self.sort(0, self._sort_order)

    def _name_matches(self, name):
        if self._regex_mode:
            try:
                return re.search(self._filter_text, name, re.IGNORECASE) is not None
            except re.error:
                # Invalid/incomplete regex (e.g. still being typed) - don't
                # blank the whole tree, just show everything until it's valid
                return True
        return self._filter_text.lower() in name.lower()

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._filter_text:
            return True
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        return self._name_matches(model.fileName(index))

    def lessThan(self, left, right):
        model = self.sourceModel()
        if self._sort_mode == "date_modified":
            return (
                model.fileInfo(left).lastModified()
                < model.fileInfo(right).lastModified()
            )
        return model.fileName(left).lower() < model.fileName(right).lower()


class FolderCountDelegate(QStyledItemDelegate):
    """Custom delegate to display folder names with image counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.show_counts = False
        self.valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".avif",
            ".tiff",
            ".bmp",
            ".tn3",
        )
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
                count = sum(
                    1
                    for f in os.listdir(path)
                    if f.lower().endswith(self.valid_extensions)
                )
                self._count_cache[path] = count
                return count
        except (PermissionError, OSError):
            pass

        return 0

    def initStyleOption(self, option, index):
        """Append image count to the folder name for every visible item."""
        super().initStyleOption(option, index)
        if not self.show_counts or index.column() != 0:
            return

        tree_view = self.parent()
        if not tree_view or not isinstance(tree_view, QTreeView):
            return

        model = tree_view.model()
        if not model:
            return

        fs_model = model.sourceModel() if hasattr(model, "sourceModel") else model
        if not isinstance(fs_model, QFileSystemModel):
            return

        src_index = model.mapToSource(index) if hasattr(model, "mapToSource") else index
        path = fs_model.filePath(src_index)

        count = self.get_image_count(path)
        if count > 0:
            option.text = f"{os.path.basename(path)} ({count})"


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

    def _file_path(self, index):
        """Return the filesystem path for index, proxy-model-aware."""
        model = self.model()
        if hasattr(model, "sourceModel"):
            return model.sourceModel().filePath(model.mapToSource(index))
        return model.filePath(index)

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
            if self.parent_app and hasattr(self.parent_app, "_settings_loaded"):
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
                if self.parent_app and hasattr(
                    self.parent_app, "on_source_folder_changed"
                ):
                    self.parent_app.on_source_folder_changed(idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_folder_prompt()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press - clear multi-selection on plain left-click."""
        # If left-click without Ctrl or Shift modifiers and we have multiple selections,
        # clear the selection first so only the clicked item gets selected
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = event.modifiers()
            has_ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
            has_shift = modifiers & Qt.KeyboardModifier.ShiftModifier

            if not has_ctrl and not has_shift:
                # Plain left-click - if we have multiple selections, clear them
                selected_indexes = self.selectedIndexes()
                # Filter to unique rows (column 0 only)
                selected_rows = set(
                    idx.row() for idx in selected_indexes if idx.column() == 0
                )
                if len(selected_rows) > 1:
                    self.clearSelection()

        super().mousePressEvent(event)

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

        dest_path = self._file_path(index)
        urls = event.mimeData().urls()
        is_copy = event.modifiers() & Qt.KeyboardModifier.ControlModifier

        # Start a new operation batch for undo tracking
        if self.parent_app:
            operation_type = "copy" if is_copy else "move"
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
            copy_path_act = QAction("Copy Path", self)
            copy_path_act.triggered.connect(lambda: self.copy_path_to_clipboard(index))
            menu.addAction(copy_path_act)
            rename_act = QAction("Rename", self)
            rename_act.triggered.connect(lambda: self.rename_prompt(index))
            menu.addAction(rename_act)
            delete_act = QAction("Delete Selection", self)
            delete_act.triggered.connect(self.delete_folder_prompt)
            menu.addAction(delete_act)
        menu.exec(self.viewport().mapToGlobal(point))

    def copy_path_to_clipboard(self, index):
        """Copy the full filesystem path of the folder to the clipboard."""
        path = self._file_path(index)
        QApplication.clipboard().setText(path)

    def create_folder_prompt(self, index):
        parent_p = (
            self._file_path(index) if index.isValid() else self.model().rootPath()
        )
        name, ok = QInputDialog.getText(
            self, "New Folder", "Folder name:", QLineEdit.EchoMode.Normal, "New Folder"
        )
        if ok and name:
            new_p = os.path.join(parent_p, name)
            if not os.path.exists(new_p):
                os.makedirs(new_p)

    def rename_prompt(self, index):
        old_path = self._file_path(index)
        new_name, ok = QInputDialog.getText(
            self,
            "Rename",
            "New name:",
            QLineEdit.EchoMode.Normal,
            os.path.basename(old_path),
        )
        if ok and new_name:
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not rename: {e}")
                return

            # This renames on disk directly rather than through the model's
            # own rename path, so QFileSystemModel's cached row for the old
            # name doesn't reliably pick up the change on its own - force
            # both trees to re-scan so the new name actually shows up.
            if self.parent_app and hasattr(self.parent_app, "refresh_directory_trees"):
                self.parent_app.refresh_directory_trees()

    def delete_folder_prompt(self):
        selected_indexes = [i for i in self.selectedIndexes() if i.column() == 0]
        if not selected_indexes:
            return
        paths = sorted([self._file_path(i) for i in selected_indexes], reverse=True)
        reply = QMessageBox.question(
            self,
            "Delete",
            f"Delete {len(paths)} folder(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for path in paths:
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete {path}: {e}")
