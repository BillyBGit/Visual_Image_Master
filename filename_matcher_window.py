"""
Filename Matcher — two independent thumbnail panes for manually pairing up
images and syncing filenames between them.

Right-clicking a thumbnail in one pane offers to rename whatever is
currently selected in the *other* pane so its filename (minus extension)
matches the right-clicked image - useful for pairing an unnamed/generic
batch of exports against a properly-named reference set.
"""

import os

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QToolButton,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QAction

from utils import find_main_window

VALID_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".avif",
    ".tiff",
    ".bmp",
    ".tn3",
}

THUMB_SIZE = 150


class MatcherPane(QWidget):
    """One path-entry + thumbnail-grid pane. Renaming acts on its sibling pane."""

    def __init__(self, side_name, status_callback=None, parent=None):
        super().__init__(parent)
        self.side_name = side_name
        self.other_pane = None
        self.other_side_name = ""
        self.status_callback = status_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(f"{side_name} folder path...")
        self.path_edit.returnPressed.connect(self.load_thumbnails)
        path_row.addWidget(self.path_edit, 1)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setMaximumWidth(30)
        self.browse_btn.setToolTip("Browse for folder")
        self.browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.browse_btn)

        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("⟳")
        self.refresh_btn.setToolTip("Refresh")
        self.refresh_btn.clicked.connect(self.load_thumbnails)
        path_row.addWidget(self.refresh_btn)

        layout.addLayout(path_row)

        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumb_list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumb_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.thumb_list.setWordWrap(True)
        self.thumb_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.thumb_list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.thumb_list, 1)

    def set_path(self, path):
        """Set the path field and load thumbnails from it."""
        self.path_edit.setText(path or "")
        self.load_thumbnails()

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self,
            f"Select {self.side_name} Folder",
            self.path_edit.text() or os.path.expanduser("~"),
        )
        if path:
            self.set_path(path)

    def load_thumbnails(self):
        """(Re)populate the thumbnail grid from the current path field."""
        self.thumb_list.clear()

        path = self.path_edit.text().strip()
        if not path or not os.path.isdir(path):
            if path and self.status_callback:
                self.status_callback(f"Not a valid folder: {path}")
            return

        try:
            files = sorted(
                f
                for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in VALID_EXTS
            )
        except OSError as e:
            if self.status_callback:
                self.status_callback(f"Could not read folder: {e}")
            return

        for f in files:
            full_path = os.path.join(path, f)
            item = QListWidgetItem(QIcon(full_path), f)
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            self.thumb_list.addItem(item)

    def _show_context_menu(self, pos):
        item = self.thumb_list.itemAt(pos)
        if not item or not self.other_pane:
            return

        menu = QMenu(self)
        rename_act = QAction(
            f"Rename Selected Image in {self.other_side_name} Pane to This Name",
            self,
        )
        rename_act.triggered.connect(lambda: self._rename_other_to(item))
        menu.addAction(rename_act)

        menu.exec(self.thumb_list.viewport().mapToGlobal(pos))

    def _rename_other_to(self, clicked_item):
        clicked_path = clicked_item.data(Qt.ItemDataRole.UserRole)
        if clicked_path and self.other_pane:
            self.other_pane.rename_selected_to(clicked_path)

    def rename_selected_to(self, name_source_path):
        """Rename this pane's selected image to match name_source_path's stem."""
        selected = self.thumb_list.selectedItems()
        if len(selected) != 1:
            QMessageBox.warning(
                self,
                "Nothing Selected",
                f"Select exactly one image in the {self.side_name} pane to rename.",
            )
            return

        old_path = selected[0].data(Qt.ItemDataRole.UserRole)
        new_stem = os.path.splitext(os.path.basename(name_source_path))[0]
        old_ext = os.path.splitext(old_path)[1]
        new_path = os.path.join(os.path.dirname(old_path), new_stem + old_ext)

        if new_path == old_path:
            if self.status_callback:
                self.status_callback(
                    f"{os.path.basename(old_path)} already has that name"
                )
            return

        if os.path.exists(new_path):
            QMessageBox.warning(
                self,
                "Rename Failed",
                f"'{os.path.basename(new_path)}' already exists in this folder.",
            )
            return

        try:
            os.rename(old_path, new_path)
        except OSError as e:
            QMessageBox.warning(self, "Rename Failed", f"Could not rename: {e}")
            return

        self.load_thumbnails()

        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == new_path:
                item.setSelected(True)
                self.thumb_list.setCurrentItem(item)
                break

        if self.status_callback:
            self.status_callback(
                f"Renamed {os.path.basename(old_path)} -> {os.path.basename(new_path)}"
            )


class FilenameMatcherWindow(QMainWindow):
    """Two-pane tool for visually pairing images and syncing filenames."""

    def __init__(self, parent=None, dir_a=None, dir_b=None):
        super().__init__(parent)

        self.parent_app = parent or find_main_window()

        self.setWindowTitle("Filename Matcher")
        self.resize(1100, 700)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        def show_status(msg):
            self.status_bar.showMessage(msg, 4000)

        self.left_pane = MatcherPane("Left", status_callback=show_status)
        self.right_pane = MatcherPane("Right", status_callback=show_status)

        self.left_pane.other_pane = self.right_pane
        self.left_pane.other_side_name = "Right"
        self.right_pane.other_pane = self.left_pane
        self.right_pane.other_side_name = "Left"

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_pane)
        splitter.addWidget(self.right_pane)
        splitter.setSizes([550, 550])
        self.setCentralWidget(splitter)

        if dir_a:
            self.left_pane.set_path(dir_a)
        if dir_b:
            self.right_pane.set_path(dir_b)
