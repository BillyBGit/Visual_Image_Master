"""
Main ImageSorter application class.
"""

import os
import shutil
from PySide6.QtWidgets import (QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
                               QLineEdit, QMessageBox, QProgressBar, QStatusBar,
                               QFileSystemModel, QToolButton, QMenu, QApplication,
                               QListWidgetItem)
from PySide6.QtCore import Qt, QSize, QDir
from PySide6.QtGui import QAction, QIcon
from PIL import Image

from thumbnail_list import ThumbnailList
from folder_tree import FolderTree
from face_detection import FaceDetector, ImageQualityAnalyzer
from settings_manager import SettingsManager


class ImageSorter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Image Master 2026")
        self.resize(1600, 900)

        # --- Persistence Placeholders ---
        self.preview_fit_default = False
        self.preview_geometry = None
        self._settings_loaded = False
        self.src_root_path = None
        self.dest_root_path = None

        # Sorting and filtering state
        self.sort_by = "name"
        self.sort_ascending = True
        self.filter_active = False
        self.filter_mode = None  # None, "has_face", "no_face"

        # Face detector
        self.face_detector = FaceDetector()

        # Image quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer()

        # Settings manager
        self.settings_manager = SettingsManager(self)

        # --- Menu Bar Setup ---
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu('&File')
        exit_act = QAction('&Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Options Menu
        options_menu = menu_bar.addMenu('&Options')

        self.remember_settings_act = QAction('Remember Settings', self, checkable=True)
        self.remember_settings_act.setChecked(True)
        options_menu.addAction(self.remember_settings_act)

        self.show_names_act = QAction('Show Filenames', self, checkable=True)
        self.show_names_act.setChecked(False)
        self.show_names_act.triggered.connect(self.toggle_filenames)
        options_menu.addAction(self.show_names_act)

        self.show_counts_act = QAction('Show Folder Counts', self, checkable=True)
        self.show_counts_act.setChecked(False)
        self.show_counts_act.triggered.connect(self.toggle_folder_counts)
        options_menu.addAction(self.show_counts_act)

        self.conf_delete_act = QAction('Confirm Deletions', self, checkable=True)
        self.conf_delete_act.setChecked(True)
        self.conf_delete_act.triggered.connect(self.toggle_conf_del)
        options_menu.addAction(self.conf_delete_act)

        # --- Status Bar and Progress Bar Setup ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # --- Main Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Source Tree (Left) with path entry
        self.src_widget = QWidget()
        self.src_layout = QVBoxLayout(self.src_widget)
        self.src_layout.setContentsMargins(0, 0, 0, 0)
        self.src_path_display = QLineEdit(os.path.expanduser("~"))
        self.src_path_display.setPlaceholderText("Source folder path...")
        self.src_path_display.returnPressed.connect(self.on_source_path_change)
        self.src_model = QFileSystemModel()
        self.src_model.setRootPath(QDir.rootPath())
        self.src_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.src_tree = FolderTree(self, is_source=True)
        self.src_tree.set_model_and_clean_columns(self.src_model)
        self.src_layout.addWidget(self.src_path_display)
        self.src_layout.addWidget(self.src_tree)

        # Thumbnail View (Middle)
        self.mid_widget = QWidget()
        self.mid_layout = QVBoxLayout(self.mid_widget)
        self.mid_layout.setContentsMargins(0, 0, 0, 0)

        # Top toolbar row with filter and sort buttons
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)

        # Filter button - handled with custom left/right click
        self.filter_btn = QToolButton()
        self.filter_btn.setText("🔽")
        self.filter_btn.setToolTip("Left-click to toggle filter, right-click for options")
        self.filter_btn.clicked.connect(self.toggle_filter)
        self.filter_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filter_btn.customContextMenuRequested.connect(self.show_filter_menu)
        self.filter_menu = QMenu(self.filter_btn)
        self.setup_filter_menu()

        # Store original button stylesheet
        self._filter_btn_original_style = self.filter_btn.styleSheet() if self.filter_btn.styleSheet() else ""

        # Sort button
        self.sort_btn = QToolButton()
        self.sort_btn.setText("⇅")
        self.sort_btn.setToolTip("Sort images")
        self.sort_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.sort_menu = QMenu(self.sort_btn)
        self.sort_btn.setMenu(self.sort_menu)
        self.setup_sort_menu()

        self.toolbar_layout.addWidget(self.filter_btn)
        self.toolbar_layout.addWidget(self.sort_btn)
        self.toolbar_layout.addStretch()

        self.thumb_list = ThumbnailList(self)
        self.mid_layout.addLayout(self.toolbar_layout)
        self.mid_layout.addWidget(self.thumb_list)

        # Destination Tree (Right) with path entry
        self.dest_widget = QWidget()
        self.dest_layout = QVBoxLayout(self.dest_widget)
        self.dest_layout.setContentsMargins(0, 0, 0, 0)
        self.dest_path_display = QLineEdit(os.path.expanduser("~"))
        self.dest_path_display.setPlaceholderText("Destination folder path...")
        self.dest_path_display.returnPressed.connect(self.on_dest_path_change)
        self.dest_model = QFileSystemModel()
        self.dest_model.setRootPath(QDir.rootPath())
        self.dest_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.dest_tree = FolderTree(self, is_source=False)
        self.dest_tree.set_model_and_clean_columns(self.dest_model)
        self.dest_layout.addWidget(self.dest_path_display)
        self.dest_layout.addWidget(self.dest_tree)

        self.main_splitter.addWidget(self.src_widget)
        self.main_splitter.addWidget(self.mid_widget)
        self.main_splitter.addWidget(self.dest_widget)
        self.setCentralWidget(self.main_splitter)

        # Connect splitter movement to trigger thumbnail redraw
        self.main_splitter.splitterMoved.connect(self.on_splitter_moved)

        # Load existing settings (blocks signals during restoration)
        self.settings_manager.read_settings()

        # Mark settings as loaded to allow subsequent writes
        self._settings_loaded = True

        # Initial Load
        self.refresh_thumbnails()

    def write_settings(self):
        """Delegate to settings manager."""
        self.settings_manager.write_settings()

    def read_settings(self):
        """Delegate to settings manager."""
        self.settings_manager.read_settings()

    def setup_sort_menu(self):
        """Create the sort menu with all sorting options."""
        name_asc_act = QAction("Name (A-Z)", self, checkable=True)
        name_asc_act.triggered.connect(lambda: self.set_sort("name", True))
        self.sort_menu.addAction(name_asc_act)

        name_desc_act = QAction("Name (Z-A)", self, checkable=True)
        name_desc_act.triggered.connect(lambda: self.set_sort("name", False))
        self.sort_menu.addAction(name_desc_act)

        self.sort_menu.addSeparator()

        modified_asc_act = QAction("Modified Date (Oldest First)", self, checkable=True)
        modified_asc_act.triggered.connect(lambda: self.set_sort("modified", True))
        self.sort_menu.addAction(modified_asc_act)

        modified_desc_act = QAction("Modified Date (Newest First)", self, checkable=True)
        modified_desc_act.triggered.connect(lambda: self.set_sort("modified", False))
        self.sort_menu.addAction(modified_desc_act)

        self.sort_menu.addSeparator()

        res_asc_act = QAction("Resolution (Low to High)", self, checkable=True)
        res_asc_act.triggered.connect(lambda: self.set_sort("resolution", True))
        self.sort_menu.addAction(res_asc_act)

        res_desc_act = QAction("Resolution (High to Low)", self, checkable=True)
        res_desc_act.triggered.connect(lambda: self.set_sort("resolution", False))
        self.sort_menu.addAction(res_desc_act)

        self.sort_menu.addSeparator()

        width_asc_act = QAction("Width (Narrow to Wide)", self, checkable=True)
        width_asc_act.triggered.connect(lambda: self.set_sort("width", True))
        self.sort_menu.addAction(width_asc_act)

        width_desc_act = QAction("Width (Wide to Narrow)", self, checkable=True)
        width_desc_act.triggered.connect(lambda: self.set_sort("width", False))
        self.sort_menu.addAction(width_desc_act)

        self.sort_menu.addSeparator()

        height_asc_act = QAction("Height (Short to Tall)", self, checkable=True)
        height_asc_act.triggered.connect(lambda: self.set_sort("height", True))
        self.sort_menu.addAction(height_asc_act)

        height_desc_act = QAction("Height (Tall to Short)", self, checkable=True)
        height_desc_act.triggered.connect(lambda: self.set_sort("height", False))
        self.sort_menu.addAction(height_desc_act)

        self.sort_menu.addSeparator()

        # Quality/Blur sorting
        quality_asc_act = QAction("Quality (Blurry to Sharp)", self, checkable=True)
        quality_asc_act.triggered.connect(lambda: self.set_sort("quality", True))
        self.sort_menu.addAction(quality_asc_act)

        quality_desc_act = QAction("Quality (Sharp to Blurry)", self, checkable=True)
        quality_desc_act.triggered.connect(lambda: self.set_sort("quality", False))
        self.sort_menu.addAction(quality_desc_act)

        self.sort_actions = {
            ("name", True): name_asc_act,
            ("name", False): name_desc_act,
            ("modified", True): modified_asc_act,
            ("modified", False): modified_desc_act,
            ("resolution", True): res_asc_act,
            ("resolution", False): res_desc_act,
            ("width", True): width_asc_act,
            ("width", False): width_desc_act,
            ("height", True): height_asc_act,
            ("height", False): height_desc_act,
            ("quality", True): quality_asc_act,
            ("quality", False): quality_desc_act,
        }

        self.sort_actions[("name", True)].setChecked(True)

    def setup_filter_menu(self):
        """Create the filter menu with face detection options."""
        self.filter_menu.clear()

        has_face_act = QAction("Contains Face", self, checkable=True)
        has_face_act.triggered.connect(lambda: self.set_filter("has_face"))
        self.filter_menu.addAction(has_face_act)

        no_face_act = QAction("Does Not Contain Face", self, checkable=True)
        no_face_act.triggered.connect(lambda: self.set_filter("no_face"))
        self.filter_menu.addAction(no_face_act)

        self.filter_actions = {
            "has_face": has_face_act,
            "no_face": no_face_act,
        }

    def show_filter_menu(self, pos):
        """Show the filter menu on right-click."""
        self.filter_menu.exec(self.filter_btn.mapToGlobal(pos))

    def toggle_filter(self):
        """Toggle filter on/off with left click."""
        self.filter_active = not self.filter_active
        self.update_filter_visuals()
        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def set_filter(self, filter_mode):
        """Set the filter mode and activate filtering."""
        if self.filter_mode == filter_mode:
            self.filter_mode = None
            self.filter_active = False
        else:
            self.filter_mode = filter_mode
            self.filter_active = True

        for mode, action in self.filter_actions.items():
            action.setChecked(mode == self.filter_mode and self.filter_active)

        self.update_filter_visuals()
        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def update_filter_visuals(self):
        """Update visual indicators for filter state."""
        if self.filter_active and self.filter_mode:
            self.filter_btn.setStyleSheet("""
                QToolButton {
                    background-color: #007FFF;
                    color: white;
                    border: 2px solid #005FCC;
                    border-radius: 3px;
                    padding: 3px;
                }
                QToolButton:hover {
                    background-color: #0066CC;
                }
            """)
            self.thumb_list.set_filter_active(True)

            filter_name = "Contains Face" if self.filter_mode == "has_face" else "Does Not Contain Face"
            self.status_bar.showMessage(f"Filter active: {filter_name}", 3000)
        else:
            if hasattr(self, '_filter_btn_original_style'):
                self.filter_btn.setStyleSheet(self._filter_btn_original_style)
            else:
                self.filter_btn.setStyleSheet("")
            self.thumb_list.set_filter_active(False)

            if not self.filter_active:
                self.status_bar.showMessage("Filter disabled", 3000)

    def set_sort(self, sort_by, ascending):
        """Update the sort criteria and refresh thumbnails."""
        self.sort_by = sort_by
        self.sort_ascending = ascending

        for key, action in self.sort_actions.items():
            action.setChecked(key == (sort_by, ascending))

        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def get_current_source_path(self):
        """Get the currently selected path in the source tree for loading thumbnails."""
        current_index = self.src_tree.currentIndex()
        if current_index.isValid():
            return self.src_model.filePath(current_index)
        return self.src_root_path if self.src_root_path else "/"

    def on_source_folder_changed(self, index):
        """Called when user selects a different folder in the source tree."""
        self.refresh_thumbnails()
        if self._settings_loaded:
            self.write_settings()

    def on_source_path_change(self):
        """Called when user manually edits the source path and presses Enter."""
        new_path = self.src_path_display.text()

        if not os.path.isdir(new_path):
            QMessageBox.warning(self, "Invalid Path",
                f"The path '{new_path}' does not exist or is not a directory.")
            if self.src_root_path and os.path.isdir(self.src_root_path):
                self.src_path_display.setText(self.src_root_path)
            return

        new_path = os.path.abspath(new_path)
        self.src_path_display.setText(new_path)

        self.src_tree.blockSignals(True)
        self.src_root_path = new_path
        self.src_tree.setRootIndex(self.src_model.index(new_path))

        index = self.src_model.index(new_path)
        if index.isValid():
            self.src_tree.setCurrentIndex(index)
            self.src_tree.scrollTo(index)

        self.src_tree.blockSignals(False)
        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def on_dest_path_change(self):
        """Called when user manually edits the destination path and presses Enter."""
        new_path = self.dest_path_display.text()

        if not os.path.isdir(new_path):
            QMessageBox.warning(self, "Invalid Path",
                f"The path '{new_path}' does not exist or is not a directory.")
            if self.dest_root_path and os.path.isdir(self.dest_root_path):
                self.dest_path_display.setText(self.dest_root_path)
            return

        new_path = os.path.abspath(new_path)
        self.dest_path_display.setText(new_path)

        self.dest_tree.blockSignals(True)
        self.dest_root_path = new_path
        self.dest_tree.setRootIndex(self.dest_model.index(new_path))

        index = self.dest_model.index(new_path)
        if index.isValid():
            self.dest_tree.setCurrentIndex(index)
            self.dest_tree.scrollTo(index)

        self.dest_tree.blockSignals(False)

        if self._settings_loaded:
            self.write_settings()

    def on_splitter_moved(self, pos, index):
        """Called when the splitter is moved - triggers thumbnail redraw."""
        self.thumb_list.scheduleDelayedItemsLayout()

        if self._settings_loaded:
            self.write_settings()

    def toggle_filenames(self):
        """Toggles visibility of text for all items in the thumbnail list."""
        show = self.show_names_act.isChecked()

        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if show:
                full_path = item.data(Qt.ItemDataRole.UserRole)
                item.setText(os.path.basename(full_path))
            else:
                item.setText("")

        if self._settings_loaded:
            self.write_settings()

    def toggle_folder_counts(self):
        """Toggles display of image counts in folder trees."""
        show = self.show_counts_act.isChecked()

        self.src_tree.set_show_counts(show)
        self.dest_tree.set_show_counts(show)

        status = "enabled" if show else "disabled"
        self.status_bar.showMessage(f"Folder counts {status}.", 3000)

        if self._settings_loaded:
            self.write_settings()

    def toggle_conf_del(self):
        state = self.conf_delete_act.isChecked()
        status = "enabled" if state else "disabled"
        self.status_bar.showMessage(f"Delete confirmation {status}.", 3000)
        if self._settings_loaded:
            self.write_settings()

    def refresh_thumbnails(self):
        self.thumb_list.clear()
        path = self.get_current_source_path()
        if not os.path.isdir(path):
            return

        valid = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif', '.tiff', '.bmp')
        files = [f for f in os.listdir(path) if f.lower().endswith(valid)]

        # Apply face filter if active
        if self.filter_active and self.filter_mode:
            filtered_files = []
            total = len(files)

            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage(f"Filtering {total} images...")

            for i, f in enumerate(files):
                full_path = os.path.join(path, f)
                has_face = self.face_detector.has_face(full_path)

                if self.filter_mode == "has_face" and has_face:
                    filtered_files.append(f)
                elif self.filter_mode == "no_face" and not has_face:
                    filtered_files.append(f)

                self.progress_bar.setValue(i + 1)
                if i % 5 == 0:
                    QApplication.processEvents()

            files = filtered_files
            self.progress_bar.setVisible(False)

        files = self.sort_files(path, files)
        total_files = len(files)

        if total_files == 0:
            if self.filter_active and self.filter_mode:
                filter_name = "with faces" if self.filter_mode == "has_face" else "without faces"
                self.status_bar.showMessage(f"No images {filter_name} found in this directory.", 3000)
            else:
                self.status_bar.showMessage("No images found in this directory.", 3000)
            return

        show_names = self.show_names_act.isChecked()

        self.progress_bar.setRange(0, total_files)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Loading {total_files} images...")

        for i, f in enumerate(files):
            full_p = os.path.join(path, f)
            display_name = f if show_names else ""

            item = QListWidgetItem(QIcon(full_p), display_name)
            item.setData(Qt.ItemDataRole.UserRole, full_p)
            self.thumb_list.addItem(item)

            self.progress_bar.setValue(i + 1)

            if i % 10 == 0:
                QApplication.processEvents()

        self.progress_bar.setVisible(False)

        if self.filter_active and self.filter_mode:
            filter_name = "with faces" if self.filter_mode == "has_face" else "without faces"
            self.status_bar.showMessage(f"Showing {total_files} images {filter_name}", 3000)
        else:
            self.status_bar.showMessage("Ready", 3000)

        if self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

    def sort_files(self, path, files):
        """Sort files based on current sort criteria."""
        if self.sort_by == "name":
            files = sorted(files, reverse=not self.sort_ascending)
        elif self.sort_by == "modified":
            files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path, f)),
                          reverse=not self.sort_ascending)
        elif self.sort_by == "resolution":
            files = sorted(files, key=lambda f: self.get_image_resolution(os.path.join(path, f)),
                          reverse=not self.sort_ascending)
        elif self.sort_by == "width":
            files = sorted(files, key=lambda f: self.get_image_width(os.path.join(path, f)),
                          reverse=not self.sort_ascending)
        elif self.sort_by == "height":
            files = sorted(files, key=lambda f: self.get_image_height(os.path.join(path, f)),
                          reverse=not self.sort_ascending)
        elif self.sort_by == "quality":
            files = sorted(files, key=lambda f: self.get_image_quality(os.path.join(path, f)),
                          reverse=not self.sort_ascending)
        return files

    def get_image_resolution(self, file_path):
        """Get total pixel count (width * height) of an image."""
        try:
            with Image.open(file_path) as img:
                return img.width * img.height
        except Exception:
            return 0

    def get_image_width(self, file_path):
        """Get width of an image."""
        try:
            with Image.open(file_path) as img:
                return img.width
        except Exception:
            return 0

    def get_image_height(self, file_path):
        """Get height of an image."""
        try:
            with Image.open(file_path) as img:
                return img.height
        except Exception:
            return 0

    def get_image_quality(self, file_path):
        """Get image quality/sharpness score."""
        return self.quality_analyzer.get_blur_score(file_path)

    def remove_thumbnails_by_path(self, paths_to_remove):
        """Removes specific items from the thumb_list without a full reload."""
        paths_to_remove = {os.path.normpath(p) for p in paths_to_remove}

        for i in range(self.thumb_list.count() - 1, -1, -1):
            item = self.thumb_list.item(i)
            item_path = os.path.normpath(item.data(Qt.ItemDataRole.UserRole))

            if item_path in paths_to_remove:
                self.thumb_list.takeItem(i)

        self.status_bar.showMessage(f"Removed {len(paths_to_remove)} items.", 3000)

        if self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

    def smart_transfer(self, src, dst_folder, is_copy):
        """Handles file transfer with automatic renaming to avoid conflicts."""
        dest = os.path.join(dst_folder, os.path.basename(src))
        name, ext = os.path.splitext(dest)
        counter = 1
        while os.path.exists(dest):
            dest = f"{name}_{counter}{ext}"
            counter += 1
        try:
            if is_copy:
                shutil.copy2(src, dest)
            else:
                shutil.move(src, dest)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not transfer file: {e}")

    def resizeEvent(self, event):
        """Handle main window resize - triggers thumbnail redraw."""
        super().resizeEvent(event)
        if hasattr(self, 'thumb_list') and hasattr(self, '_settings_loaded'):
            self.thumb_list.scheduleDelayedItemsLayout()

            if self._settings_loaded:
                self.write_settings()

    def closeEvent(self, event):
        """Save settings when the application closes."""
        self.write_settings()
        event.accept()
