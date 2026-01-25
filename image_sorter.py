"""
Main ImageSorter application class.
"""

import os
import shutil
import tempfile
import atexit
from PySide6.QtWidgets import (QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
                               QLineEdit, QMessageBox, QProgressBar, QStatusBar,
                               QFileSystemModel, QToolButton, QMenu, QApplication,
                               QListWidgetItem, QPushButton, QFileDialog, QDialog,
                               QTextBrowser, QLabel, QDialogButtonBox, QScrollArea,
                               QProgressDialog)
from PySide6.QtCore import Qt, QSize, QDir, QTimer
from PySide6.QtGui import QAction, QIcon, QActionGroup
from PIL import Image


class DeleteUndoManager:
    """Manages undo functionality for file deletions.

    Stores deleted files in a temporary directory for potential restoration.
    Only keeps track of the last deleted file(s) - single depth undo.
    """

    def __init__(self):
        # Create a temp directory for storing deleted files
        self.temp_dir = tempfile.mkdtemp(prefix="img_sort_undo_")
        # Store info about the last deletion: list of (original_path, temp_path) tuples
        self.last_deletion = []
        # Register cleanup on exit
        atexit.register(self.cleanup)

    def store_for_undo(self, file_path):
        """Move a file to temp storage for potential undo.

        Args:
            file_path: Path to the file to be "deleted"

        Returns:
            True if file was successfully stored, False otherwise
        """
        if not os.path.exists(file_path):
            return False

        try:
            # Generate a unique temp filename preserving extension
            basename = os.path.basename(file_path)
            temp_path = os.path.join(self.temp_dir, basename)

            # Handle naming conflicts in temp dir
            counter = 1
            name, ext = os.path.splitext(temp_path)
            while os.path.exists(temp_path):
                temp_path = f"{name}_{counter}{ext}"
                counter += 1

            # Move file to temp location
            shutil.move(file_path, temp_path)

            # Store the mapping
            self.last_deletion.append((file_path, temp_path))
            return True

        except Exception as e:
            print(f"Error storing file for undo: {e}")
            return False

    def start_new_deletion(self):
        """Clear previous deletion info before starting a new deletion batch.

        Call this before a new delete operation to ensure only the latest
        deletion can be undone.
        """
        # Clean up old temp files from previous deletion
        for original_path, temp_path in self.last_deletion:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_path}: {e}")

        self.last_deletion = []

    def permanent_delete(self, file_path):
        """Permanently delete a file without undo capability.

        Args:
            file_path: Path to the file to delete permanently

        Returns:
            True if file was successfully deleted, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error permanently deleting file: {e}")
            return False

    def undo(self):
        """Restore the last deleted file(s).

        Returns:
            List of restored file paths, or empty list if nothing to undo
        """
        if not self.last_deletion:
            return []

        restored_paths = []
        for original_path, temp_path in self.last_deletion:
            try:
                if os.path.exists(temp_path):
                    # Ensure the original directory still exists
                    original_dir = os.path.dirname(original_path)
                    if not os.path.exists(original_dir):
                        os.makedirs(original_dir)

                    # Move file back to original location
                    shutil.move(temp_path, original_path)
                    restored_paths.append(original_path)
            except Exception as e:
                print(f"Error restoring file {original_path}: {e}")

        self.last_deletion = []
        return restored_paths

    def can_undo(self):
        """Check if there's a deletion that can be undone."""
        return len(self.last_deletion) > 0

    def cleanup(self):
        """Clean up the temp directory and all its contents."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")

from thumbnail_list import ThumbnailList
from folder_tree import FolderTree
from face_detection import FaceDetector, ImageQualityAnalyzer
from face_detection_dialog import FaceDetectionDialog
from advanced_filter_dialog import AdvancedFilterDialog
from object_detection import get_detector as get_object_detector
from quality_assessment import get_assessor as get_quality_assessor
from scene_recognition import get_recognizer as get_scene_recognizer
from settings_manager import SettingsManager
from yolo_window import YoloWindow


class ImageSorter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Image Master 2026")
        self.resize(1600, 900)

        # --- Persistence Placeholders ---
        self.preview_fit_default = False
        self.preview_autoname_extract = False
        self.preview_geometry = None
        self.yolo_geometry = None
        self.yolo_labels = []
        self.yolo_output_path = ""
        self.yolo_resolution_index = 0
        self.yolo_size_mode = "image"
        self.caption_backend = "florence2"
        self.caption_ollama_url = "http://localhost:11434"
        self.caption_ollama_model = "llava"
        self.caption_florence_model = "base"
        self.caption_florence_detail = "detailed"
        self.caption_florence_custom_path = ""
        self.caption_wd14_model = "vit-v2"
        self.caption_wd14_threshold = 0.55  # Higher threshold to reduce excessive tags
        self.caption_wd14_custom_paths = {}  # Dict of model_key -> custom path
        self.caption_blend_instructions = ""  # Custom Ollama blending instructions
        self.caption_blend_model = ""  # Ollama model to use for blending
        self._settings_loaded = False
        self.src_root_path = None
        self.dest_root_path = None

        # Sorting and filtering state
        self.sort_by = "name"
        self.sort_ascending = True
        self.filter_active = False
        self.filter_mode = None  # None, "has_face", "no_face", "advanced"
        self.advanced_filter_settings = {}  # Settings from advanced filter dialog

        # Face detection preset
        self.face_detection_preset = "balanced"

        # Face detector (will be initialized with preset after settings load)
        self.face_detector = None

        # Image quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer()

        # Advanced AI detectors (lazy-loaded)
        self._object_detector = None
        self._quality_assessor = None
        self._scene_recognizer = None

        # Undo state for move/copy operations
        self.last_operation = None  # {"type": "move"|"copy", "files": [(src, dest), ...]}

        # Delete undo manager (for restoring deleted files)
        self.delete_undo_manager = DeleteUndoManager()

        # Deduplication mode for thumbnail pane
        self.dedup_mode = "exact"  # "exact", "visual_largest", "visual_smallest"

        # Folder deduplication settings
        self.folder_dedup_mode = "exact"  # "exact", "visual_largest", "visual_smallest", "visual_deepest", "visual_shallow"
        self.folder_dedup_recursive = False  # Whether to process subdirectories

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

        self.show_dimensions_act = QAction('Show Dimensions', self, checkable=True)
        self.show_dimensions_act.setChecked(False)
        self.show_dimensions_act.triggered.connect(self.toggle_dimensions)
        options_menu.addAction(self.show_dimensions_act)

        self.show_filesize_act = QAction('Show File Size', self, checkable=True)
        self.show_filesize_act.setChecked(False)
        self.show_filesize_act.triggered.connect(self.toggle_filesize)
        options_menu.addAction(self.show_filesize_act)

        self.show_counts_act = QAction('Show Folder Counts', self, checkable=True)
        self.show_counts_act.setChecked(False)
        self.show_counts_act.triggered.connect(self.toggle_folder_counts)
        options_menu.addAction(self.show_counts_act)

        self.conf_delete_act = QAction('Confirm Deletions', self, checkable=True)
        self.conf_delete_act.setChecked(True)
        self.conf_delete_act.triggered.connect(self.toggle_conf_del)
        options_menu.addAction(self.conf_delete_act)

        options_menu.addSeparator()

        face_detect_settings_act = QAction('Face Detection Settings...', self)
        face_detect_settings_act.triggered.connect(self.show_face_detection_settings)
        options_menu.addAction(face_detect_settings_act)

        # Tools Menu
        tools_menu = menu_bar.addMenu('&Tools')

        yolo_labeling_act = QAction('&YOLO Labelling...', self)
        yolo_labeling_act.triggered.connect(self.open_yolo_window)
        tools_menu.addAction(yolo_labeling_act)

        auto_caption_act = QAction('&Auto Caption...', self)
        auto_caption_act.triggered.connect(self.open_caption_window)
        tools_menu.addAction(auto_caption_act)

        # Help Menu
        help_menu = menu_bar.addMenu('&Help')

        manual_act = QAction('&Manual', self)
        manual_act.setShortcut('F1')
        manual_act.triggered.connect(self.show_manual)
        help_menu.addAction(manual_act)

        help_menu.addSeparator()

        about_act = QAction('&About', self)
        about_act.triggered.connect(self.show_about)
        help_menu.addAction(about_act)

        # --- Status Bar and Progress Bar Setup ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # --- Main Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Source Tree (Left) with path entry and browse button
        self.src_widget = QWidget()
        self.src_layout = QVBoxLayout(self.src_widget)
        self.src_layout.setContentsMargins(0, 0, 0, 0)

        # Source path row with text entry, browse button, and dedup button
        self.src_path_row = QHBoxLayout()
        self.src_path_display = QLineEdit(os.path.expanduser("~"))
        self.src_path_display.setPlaceholderText("Source folder path...")
        self.src_path_display.returnPressed.connect(self.on_source_path_change)
        self.src_browse_btn = QPushButton("...")
        self.src_browse_btn.setMaximumWidth(30)
        self.src_browse_btn.setToolTip("Browse for source folder")
        self.src_browse_btn.clicked.connect(self.browse_source_folder)
        self.src_dedup_btn = QPushButton("🗑️")
        self.src_dedup_btn.setMaximumWidth(30)
        self.src_dedup_btn.setToolTip("Remove duplicates in selected folders (right-click for options)")
        self.src_dedup_btn.clicked.connect(lambda: self.remove_folder_duplicates(self.src_tree))
        self.src_dedup_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.src_dedup_btn.customContextMenuRequested.connect(self.show_folder_dedup_menu)
        self.src_path_row.addWidget(self.src_path_display)
        self.src_path_row.addWidget(self.src_browse_btn)
        self.src_path_row.addWidget(self.src_dedup_btn)

        self.src_model = QFileSystemModel()
        self.src_model.setRootPath(QDir.rootPath())
        self.src_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.src_tree = FolderTree(self, is_source=True)
        self.src_tree.set_model_and_clean_columns(self.src_model)
        self.src_layout.addLayout(self.src_path_row)
        self.src_layout.addWidget(self.src_tree)

        # Thumbnail View (Middle)
        self.mid_widget = QWidget()
        self.mid_layout = QVBoxLayout(self.mid_widget)
        self.mid_layout.setContentsMargins(0, 0, 0, 0)

        # Top toolbar row with filter and sort buttons
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)

        # Set icon size for toolbar buttons (30% larger)
        icon_size = QSize(32, 32)  # Increased from default ~24x24

        # Filter button - handled with custom left/right click
        self.filter_btn = QToolButton()
        self.filter_btn.setText("🔽")
        self.filter_btn.setIconSize(icon_size)
        self.filter_btn.setMinimumSize(icon_size)
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
        self.sort_btn.setIconSize(icon_size)
        self.sort_btn.setMinimumSize(icon_size)
        self.sort_btn.setToolTip("Sort images")
        self.sort_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.sort_menu = QMenu(self.sort_btn)
        self.sort_btn.setMenu(self.sort_menu)
        self.setup_sort_menu()

        # Deduplicate button
        self.dedup_btn = QToolButton()
        self.dedup_btn.setText("🗑️")
        self.dedup_btn.setIconSize(icon_size)
        self.dedup_btn.setMinimumSize(icon_size)
        self.dedup_btn.setToolTip("Remove duplicates (right-click for options)")
        self.dedup_btn.clicked.connect(self.remove_duplicates)
        self.dedup_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dedup_btn.customContextMenuRequested.connect(self.show_dedup_menu)
        self.dedup_menu = QMenu(self.dedup_btn)
        self.setup_dedup_menu()

        # Batch caption button
        self.batch_caption_btn = QToolButton()
        self.batch_caption_btn.setText("📝")
        self.batch_caption_btn.setIconSize(icon_size)
        self.batch_caption_btn.setMinimumSize(icon_size)
        self.batch_caption_btn.setToolTip("Generate captions for images (selected or all)")
        self.batch_caption_btn.clicked.connect(self.batch_generate_captions)

        self.toolbar_layout.addWidget(self.filter_btn)
        self.toolbar_layout.addWidget(self.sort_btn)
        self.toolbar_layout.addWidget(self.dedup_btn)
        self.toolbar_layout.addWidget(self.batch_caption_btn)
        self.toolbar_layout.addStretch()

        self.thumb_list = ThumbnailList(self)
        self.mid_layout.addLayout(self.toolbar_layout)
        self.mid_layout.addWidget(self.thumb_list)

        # Destination Tree (Right) with path entry and browse button
        self.dest_widget = QWidget()
        self.dest_layout = QVBoxLayout(self.dest_widget)
        self.dest_layout.setContentsMargins(0, 0, 0, 0)

        # Destination path row with text entry, browse button, and dedup button
        self.dest_path_row = QHBoxLayout()
        self.dest_path_display = QLineEdit(os.path.expanduser("~"))
        self.dest_path_display.setPlaceholderText("Destination folder path...")
        self.dest_path_display.returnPressed.connect(self.on_dest_path_change)
        self.dest_browse_btn = QPushButton("...")
        self.dest_browse_btn.setMaximumWidth(30)
        self.dest_browse_btn.setToolTip("Browse for destination folder")
        self.dest_browse_btn.clicked.connect(self.browse_dest_folder)
        self.dest_dedup_btn = QPushButton("🗑️")
        self.dest_dedup_btn.setMaximumWidth(30)
        self.dest_dedup_btn.setToolTip("Remove duplicates in selected folders (right-click for options)")
        self.dest_dedup_btn.clicked.connect(lambda: self.remove_folder_duplicates(self.dest_tree))
        self.dest_dedup_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dest_dedup_btn.customContextMenuRequested.connect(self.show_folder_dedup_menu)
        self.dest_path_row.addWidget(self.dest_path_display)
        self.dest_path_row.addWidget(self.dest_browse_btn)
        self.dest_path_row.addWidget(self.dest_dedup_btn)

        self.dest_model = QFileSystemModel()
        self.dest_model.setRootPath(QDir.rootPath())
        self.dest_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.dest_tree = FolderTree(self, is_source=False)
        self.dest_tree.set_model_and_clean_columns(self.dest_model)
        self.dest_layout.addLayout(self.dest_path_row)
        self.dest_layout.addWidget(self.dest_tree)

        self.main_splitter.addWidget(self.src_widget)
        self.main_splitter.addWidget(self.mid_widget)
        self.main_splitter.addWidget(self.dest_widget)
        self.setCentralWidget(self.main_splitter)

        # Connect splitter movement to trigger thumbnail redraw
        self.main_splitter.splitterMoved.connect(self.on_splitter_moved)

        # Load existing settings (blocks signals during restoration)
        self.settings_manager.read_settings()

        # Load WD14 custom paths into caption generator
        if self.caption_wd14_custom_paths:
            from caption_generator import get_caption_generator
            generator = get_caption_generator()
            for model_key, path in self.caption_wd14_custom_paths.items():
                generator.set_wd14_custom_path(model_key, path)

        # Initialize face detector with loaded preset
        self.face_detector = FaceDetector(self.face_detection_preset)

        # Mark settings as loaded to allow subsequent writes
        self._settings_loaded = True

        # Initial Load
        self.refresh_thumbnails()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for main window."""
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.undo_last_operation()
        else:
            super().keyPressEvent(event)

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

        self.sort_menu.addSeparator()

        # File size sorting
        size_asc_act = QAction("File Size (Smallest First)", self, checkable=True)
        size_asc_act.triggered.connect(lambda: self.set_sort("filesize", True))
        self.sort_menu.addAction(size_asc_act)

        size_desc_act = QAction("File Size (Largest First)", self, checkable=True)
        size_desc_act.triggered.connect(lambda: self.set_sort("filesize", False))
        self.sort_menu.addAction(size_desc_act)

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
            ("filesize", True): size_asc_act,
            ("filesize", False): size_desc_act,
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

        self.filter_menu.addSeparator()

        advanced_act = QAction("Advanced AI Filters...", self)
        advanced_act.triggered.connect(self.show_advanced_filters)
        self.filter_menu.addAction(advanced_act)

        self.filter_menu.addSeparator()

        # Invert filter option
        self.invert_filter_act = QAction("Invert Filter", self, checkable=True)
        self.invert_filter_act.triggered.connect(self.toggle_filter_inversion)
        self.filter_menu.addAction(self.invert_filter_act)

        self.filter_actions = {
            "has_face": has_face_act,
            "no_face": no_face_act,
        }

    def setup_dedup_menu(self):
        """Create the deduplication menu with different modes."""
        self.dedup_menu.clear()

        # Create action group for mutual exclusivity
        dedup_group = QActionGroup(self)
        dedup_group.setExclusive(True)

        exact_act = QAction("Exact (MD5 checksum)", self, checkable=True)
        exact_act.triggered.connect(lambda: self.set_dedup_mode("exact"))
        dedup_group.addAction(exact_act)
        self.dedup_menu.addAction(exact_act)

        visual_largest_act = QAction("Visual - Retain Largest", self, checkable=True)
        visual_largest_act.triggered.connect(lambda: self.set_dedup_mode("visual_largest"))
        dedup_group.addAction(visual_largest_act)
        self.dedup_menu.addAction(visual_largest_act)

        visual_smallest_act = QAction("Visual - Retain Smallest", self, checkable=True)
        visual_smallest_act.triggered.connect(lambda: self.set_dedup_mode("visual_smallest"))
        dedup_group.addAction(visual_smallest_act)
        self.dedup_menu.addAction(visual_smallest_act)

        # Set default checked
        exact_act.setChecked(True)

        self.dedup_actions = {
            "exact": exact_act,
            "visual_largest": visual_largest_act,
            "visual_smallest": visual_smallest_act,
        }

    def set_dedup_mode(self, mode):
        """Set the deduplication mode."""
        self.dedup_mode = mode
        self.dedup_actions[mode].setChecked(True)

    def show_dedup_menu(self, pos):
        """Show the deduplication menu on right-click."""
        self.dedup_menu.exec(self.dedup_btn.mapToGlobal(pos))

    def setup_folder_dedup_menu(self):
        """Create the folder deduplication menu with extended options."""
        self.folder_dedup_menu = QMenu(self)

        # Create action group for mutual exclusivity of modes
        folder_dedup_group = QActionGroup(self)
        folder_dedup_group.setExclusive(True)

        exact_act = QAction("Exact (MD5 checksum)", self, checkable=True)
        exact_act.triggered.connect(lambda: self.set_folder_dedup_mode("exact"))
        folder_dedup_group.addAction(exact_act)
        self.folder_dedup_menu.addAction(exact_act)

        visual_largest_act = QAction("Visual - Retain Largest", self, checkable=True)
        visual_largest_act.triggered.connect(lambda: self.set_folder_dedup_mode("visual_largest"))
        folder_dedup_group.addAction(visual_largest_act)
        self.folder_dedup_menu.addAction(visual_largest_act)

        visual_smallest_act = QAction("Visual - Retain Smallest", self, checkable=True)
        visual_smallest_act.triggered.connect(lambda: self.set_folder_dedup_mode("visual_smallest"))
        folder_dedup_group.addAction(visual_smallest_act)
        self.folder_dedup_menu.addAction(visual_smallest_act)

        visual_deepest_act = QAction("Visual - Retain Deepest", self, checkable=True)
        visual_deepest_act.triggered.connect(lambda: self.set_folder_dedup_mode("visual_deepest"))
        folder_dedup_group.addAction(visual_deepest_act)
        self.folder_dedup_menu.addAction(visual_deepest_act)

        visual_shallow_act = QAction("Visual - Retain Shallow", self, checkable=True)
        visual_shallow_act.triggered.connect(lambda: self.set_folder_dedup_mode("visual_shallow"))
        folder_dedup_group.addAction(visual_shallow_act)
        self.folder_dedup_menu.addAction(visual_shallow_act)

        # Separator before recursive option
        self.folder_dedup_menu.addSeparator()

        # Recursive toggle (independent, not mutually exclusive)
        self.folder_recursive_act = QAction("Recursive (include subdirectories)", self, checkable=True)
        self.folder_recursive_act.triggered.connect(self.toggle_folder_dedup_recursive)
        self.folder_dedup_menu.addAction(self.folder_recursive_act)

        # Set default checked
        exact_act.setChecked(True)

        self.folder_dedup_actions = {
            "exact": exact_act,
            "visual_largest": visual_largest_act,
            "visual_smallest": visual_smallest_act,
            "visual_deepest": visual_deepest_act,
            "visual_shallow": visual_shallow_act,
        }

    def set_folder_dedup_mode(self, mode):
        """Set the folder deduplication mode."""
        self.folder_dedup_mode = mode
        self.folder_dedup_actions[mode].setChecked(True)

    def toggle_folder_dedup_recursive(self):
        """Toggle recursive folder deduplication."""
        self.folder_dedup_recursive = self.folder_recursive_act.isChecked()

    def show_folder_dedup_menu(self, pos):
        """Show the folder deduplication menu on right-click."""
        # Ensure menu is set up
        if not hasattr(self, 'folder_dedup_menu'):
            self.setup_folder_dedup_menu()
        # Get the sender button
        sender = self.sender()
        if sender:
            self.folder_dedup_menu.exec(sender.mapToGlobal(pos))

    def show_filter_menu(self, pos):
        """Show the filter menu on right-click."""
        # Enable/disable invert filter based on whether a filter is active
        filter_is_active = self.filter_active and self.filter_mode is not None
        self.invert_filter_act.setEnabled(filter_is_active)

        # Update invert filter checkmark based on current advanced filter settings
        if self.filter_mode == "advanced":
            is_inverted = self.advanced_filter_settings.get('global_invert', False)
            self.invert_filter_act.setChecked(is_inverted)
        else:
            self.invert_filter_act.setChecked(False)

        self.filter_menu.exec(self.filter_btn.mapToGlobal(pos))

    def toggle_filter_inversion(self):
        """Toggle the global filter inversion for advanced filters."""
        if self.filter_mode == "advanced":
            # Toggle the global_invert flag
            current_invert = self.advanced_filter_settings.get('global_invert', False)
            self.advanced_filter_settings['global_invert'] = not current_invert

            # Refresh thumbnails to apply the inverted filter
            self.refresh_thumbnails()

            if self._settings_loaded:
                self.write_settings()
        else:
            QMessageBox.information(
                self,
                "Advanced Filters Only",
                "Filter inversion only applies to Advanced AI Filters.\n\n"
                "Please activate Advanced AI Filters first by right-clicking "
                "the filter button and selecting 'Advanced AI Filters...'"
            )

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

            if self.filter_mode == "has_face":
                filter_name = "Contains Face"
            elif self.filter_mode == "no_face":
                filter_name = "Does Not Contain Face"
            else:
                filter_name = "Advanced AI Filters"
            self.status_bar.showMessage(f"Filter active: {filter_name}", 3000)
        else:
            if hasattr(self, '_filter_btn_original_style'):
                self.filter_btn.setStyleSheet(self._filter_btn_original_style)
            else:
                self.filter_btn.setStyleSheet("")
            self.thumb_list.set_filter_active(False)

            if not self.filter_active:
                self.status_bar.showMessage("Filter disabled", 3000)

    def show_advanced_filters(self):
        """Show the advanced AI filters dialog."""
        dialog = AdvancedFilterDialog(self)

        # Pre-populate with current settings if advanced mode was active
        if self.filter_mode == "advanced" and self.advanced_filter_settings:
            dialog.set_filter_settings(self.advanced_filter_settings)

        if dialog.exec():
            self.advanced_filter_settings = dialog.get_filter_settings()

            # Debug output
            print(f"Advanced filter settings: {self.advanced_filter_settings}")
            print(f"Has active filters: {dialog.has_active_filters()}")

            if dialog.has_active_filters():
                self.filter_mode = "advanced"
                self.filter_active = True

                # Reset debug counter
                if hasattr(self, '_filter_debug_count'):
                    delattr(self, '_filter_debug_count')

                # Uncheck simple filter actions
                for action in self.filter_actions.values():
                    action.setChecked(False)

                self.update_filter_visuals()
                self.refresh_thumbnails()
                print(f"Applied advanced filters - mode: {self.filter_mode}, active: {self.filter_active}")
            else:
                # No filters enabled, clear advanced mode
                self.filter_mode = None
                self.filter_active = False
                self.advanced_filter_settings = {}
                self.update_filter_visuals()
                self.refresh_thumbnails()

            if self._settings_loaded:
                self.write_settings()

    def get_advanced_detectors(self):
        """Lazy-load and return advanced AI detectors."""
        if self._object_detector is None:
            conf = self.advanced_filter_settings.get('object_confidence', 0.5)
            self._object_detector = get_object_detector(conf)

            # Set custom model if specified
            custom_model_path = self.advanced_filter_settings.get('custom_model_path')
            if custom_model_path:
                try:
                    self._object_detector.set_custom_model(custom_model_path)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Custom Model Error",
                        f"Failed to load custom YOLO model:\n{e}\n\nFalling back to default model."
                    )

        if self._quality_assessor is None:
            self._quality_assessor = get_quality_assessor()

        if self._scene_recognizer is None:
            conf = self.advanced_filter_settings.get('scene_confidence', 0.3)
            self._scene_recognizer = get_scene_recognizer(conf)

        return self._object_detector, self._quality_assessor, self._scene_recognizer

    def matches_advanced_filters(self, image_path):
        """Check if image matches advanced filter criteria."""
        if not self.advanced_filter_settings:
            return True

        obj_detector, quality_assessor, scene_recognizer = self.get_advanced_detectors()

        # Debug: Track which filters are being evaluated
        if not hasattr(self, '_filter_debug_count'):
            self._filter_debug_count = 0
            print(f"\n=== Starting advanced filter evaluation ===")
            print(f"Filter settings: {self.advanced_filter_settings}")

        # Face detection filter
        if self.advanced_filter_settings.get('face_detection', False):
            min_faces = self.advanced_filter_settings.get('face_count_min', 0)
            face_count = len(self.face_detector.detect_faces(image_path))
            inverse = self.advanced_filter_settings.get('face_inverse', False)

            if inverse:
                # Does NOT contain: face count must be less than minimum
                if face_count >= min_faces:
                    return False
            else:
                # Contains: face count must be at least minimum
                if face_count < min_faces:
                    return False

        # Object detection filter
        if self.advanced_filter_settings.get('object_detection', False):
            categories = self.advanced_filter_settings.get('object_categories', [])
            inverse = self.advanced_filter_settings.get('object_inverse', False)
            custom_model_path = self.advanced_filter_settings.get('custom_model_path')

            self._filter_debug_count += 1
            if self._filter_debug_count <= 3:  # Debug first 3 images
                print(f"\n[Image {self._filter_debug_count}] {os.path.basename(image_path)}")
                print(f"  Categories: {categories}")
                print(f"  Inverse: {inverse}")
                print(f"  Custom model: {custom_model_path}")

            # For custom models: if no categories selected, check for ANY detection
            if custom_model_path and not categories:
                detections = obj_detector.detect_objects(image_path)
                has_detections = len(detections) > 0

                if self._filter_debug_count <= 3:
                    print(f"  Custom model detections: {detections}")

                if inverse:
                    # Does NOT contain: must have no detections
                    if has_detections:
                        if self._filter_debug_count <= 3:
                            print(f"  REJECTED: has detections (inverse mode)")
                        return False
                else:
                    # Contains: must have at least one detection
                    if not has_detections:
                        if self._filter_debug_count <= 3:
                            print(f"  REJECTED: no detections")
                        return False
            # For standard models or when categories are selected
            elif categories:
                matches_any = False
                for category in categories:
                    if obj_detector.has_category(image_path, category):
                        matches_any = True
                        if self._filter_debug_count <= 3:
                            print(f"  Found category: {category}")
                        break

                if self._filter_debug_count <= 3:
                    print(f"  Matches any category: {matches_any}")

                if inverse:
                    # Does NOT contain: must not match any category
                    if matches_any:
                        if self._filter_debug_count <= 3:
                            print(f"  REJECTED: matches category (inverse mode)")
                        return False
                else:
                    # Contains: must match at least one category
                    if not matches_any:
                        if self._filter_debug_count <= 3:
                            print(f"  REJECTED: no category match")
                        return False

            if self._filter_debug_count <= 3:
                print(f"  PASSED object detection filter")
            # If no categories selected and no custom model, skip this filter (don't exclude anything)

        # Scene recognition filter
        if self.advanced_filter_settings.get('scene_recognition', False):
            scene_groups = self.advanced_filter_settings.get('scene_groups', [])
            inverse = self.advanced_filter_settings.get('scene_inverse', False)

            # Only apply filter if scene groups are selected
            if scene_groups:
                matches_any = False
                for group in scene_groups:
                    if scene_recognizer.has_scene_group(image_path, group):
                        matches_any = True
                        break

                if inverse:
                    # Does NOT contain: must not match any scene group
                    if matches_any:
                        return False
                else:
                    # Contains: must match at least one scene group
                    if not matches_any:
                        return False
            # If no scene groups selected, skip this filter (don't exclude anything)

        # Quality filter
        if self.advanced_filter_settings.get('quality_filter', False):
            quality = quality_assessor.assess_quality(image_path)
            inverse = self.advanced_filter_settings.get('quality_inverse', False)

            if quality is None:
                # If we can't assess quality, treat as not meeting criteria
                return inverse  # Return True if inverse (looking for bad quality), False otherwise

            min_score = self.advanced_filter_settings.get('quality_min_score', 60)
            min_sharpness = self.advanced_filter_settings.get('quality_sharpness_min', 40)
            min_bright = self.advanced_filter_settings.get('quality_brightness_min', 50)
            max_bright = self.advanced_filter_settings.get('quality_brightness_max', 200)

            meets_criteria = (
                quality['overall_score'] >= min_score and
                quality['sharpness'] >= min_sharpness and
                min_bright <= quality['brightness'] <= max_bright
            )

            if inverse:
                # Does NOT meet: must fail criteria
                if meets_criteria:
                    return False
            else:
                # Meets: must pass criteria
                if not meets_criteria:
                    return False

        # File size filter
        if self.advanced_filter_settings.get('filesize_filter', False):
            min_kb = self.advanced_filter_settings.get('filesize_min_kb', 0)
            max_kb = self.advanced_filter_settings.get('filesize_max_kb', 10240)

            try:
                file_size_bytes = os.path.getsize(image_path)
                file_size_kb = file_size_bytes / 1024

                # Check if file size is within range
                if file_size_kb < min_kb or file_size_kb > max_kb:
                    return False
            except Exception:
                # If we can't get file size, exclude the file
                return False

        # Apply global invert if enabled
        result = True
        if self.advanced_filter_settings.get('global_invert', False):
            result = not result

        return result

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

    def browse_source_folder(self):
        """Open dialog to browse for source folder."""
        current_path = self.src_path_display.text() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", current_path
        )
        if folder and os.path.isdir(folder):
            folder = os.path.abspath(folder)
            self.src_path_display.setText(folder)

            # Directly update the tree instead of calling on_source_path_change
            self.src_tree.blockSignals(True)
            self.src_root_path = folder
            self.src_tree.setRootIndex(self.src_model.index(folder))
            index = self.src_model.index(folder)
            if index.isValid():
                self.src_tree.setCurrentIndex(index)
                self.src_tree.scrollTo(index)
            self.src_tree.blockSignals(False)
            self.refresh_thumbnails()

            if self._settings_loaded:
                self.write_settings()

    def browse_dest_folder(self):
        """Open dialog to browse for destination folder."""
        current_path = self.dest_path_display.text() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", current_path
        )
        if folder and os.path.isdir(folder):
            folder = os.path.abspath(folder)
            self.dest_path_display.setText(folder)

            # Directly update the tree instead of calling on_dest_path_change
            self.dest_tree.blockSignals(True)
            self.dest_root_path = folder
            self.dest_tree.setRootIndex(self.dest_model.index(folder))
            index = self.dest_model.index(folder)
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
        """Toggles visibility of filenames for all items in the thumbnail list."""
        show = self.show_names_act.isChecked()

        # If enabling filenames, disable dimensions and filesize (mutually exclusive)
        if show:
            if self.show_dimensions_act.isChecked():
                self.show_dimensions_act.setChecked(False)
            if self.show_filesize_act.isChecked():
                self.show_filesize_act.setChecked(False)

        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if show:
                full_path = item.data(Qt.ItemDataRole.UserRole)
                item.setText(os.path.basename(full_path))
            else:
                item.setText("")

        if self._settings_loaded:
            self.write_settings()

    def toggle_dimensions(self):
        """Toggles visibility of image dimensions for all items in the thumbnail list."""
        show = self.show_dimensions_act.isChecked()

        # If enabling dimensions, disable filenames and filesize (mutually exclusive)
        if show:
            if self.show_names_act.isChecked():
                self.show_names_act.setChecked(False)
            if self.show_filesize_act.isChecked():
                self.show_filesize_act.setChecked(False)

        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if show:
                full_path = item.data(Qt.ItemDataRole.UserRole)
                dimensions = self.get_image_dimensions_str(full_path)
                item.setText(dimensions)
            else:
                item.setText("")

        if self._settings_loaded:
            self.write_settings()

    def toggle_filesize(self):
        """Toggles visibility of file size for all items in the thumbnail list."""
        show = self.show_filesize_act.isChecked()

        # If enabling filesize, disable filenames and dimensions (mutually exclusive)
        if show:
            if self.show_names_act.isChecked():
                self.show_names_act.setChecked(False)
            if self.show_dimensions_act.isChecked():
                self.show_dimensions_act.setChecked(False)

        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            if show:
                full_path = item.data(Qt.ItemDataRole.UserRole)
                filesize = self.get_file_size_str(full_path)
                item.setText(filesize)
            else:
                item.setText("")

        if self._settings_loaded:
            self.write_settings()

    def get_image_dimensions_str(self, file_path):
        """Get image dimensions as a formatted string (widthxheight)."""
        try:
            with Image.open(file_path) as img:
                return f"{img.width}x{img.height}"
        except Exception:
            return "?x?"

    def get_file_size_str(self, file_path):
        """Get file size as a human-readable string."""
        try:
            size_bytes = os.path.getsize(file_path)
            # Convert to human-readable format
            if size_bytes < 1024:
                return f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f}KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f}MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"
        except Exception:
            return "?"

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

    def show_face_detection_settings(self):
        """Show face detection settings dialog."""
        dialog = FaceDetectionDialog(self, self.face_detection_preset)

        if dialog.exec():
            new_preset = dialog.get_selected_preset()

            if new_preset != self.face_detection_preset:
                self.face_detection_preset = new_preset
                self.face_detector.set_preset(new_preset)

                self.status_bar.showMessage(
                    f"Face detection preset changed to: {new_preset.capitalize()}",
                    5000
                )

                if self._settings_loaded:
                    self.write_settings()

    def refresh_thumbnails(self):
        # Show progress bar immediately
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.status_bar.showMessage("Loading directory...")
        QApplication.processEvents()  # Force UI update

        self.thumb_list.clear()
        path = self.get_current_source_path()
        if not os.path.isdir(path):
            self.progress_bar.setVisible(False)
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

                # Apply appropriate filter
                include = False
                if self.filter_mode == "has_face":
                    include = self.face_detector.has_face(full_path)
                elif self.filter_mode == "no_face":
                    include = not self.face_detector.has_face(full_path)
                elif self.filter_mode == "advanced":
                    include = self.matches_advanced_filters(full_path)

                if include:
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
                if self.filter_mode == "has_face":
                    filter_name = "with faces"
                elif self.filter_mode == "no_face":
                    filter_name = "without faces"
                else:
                    filter_name = "matching AI filters"
                self.status_bar.showMessage(f"No images {filter_name} found in this directory.", 3000)
            else:
                self.status_bar.showMessage("No images found in this directory.", 3000)
            return

        show_names = self.show_names_act.isChecked()
        show_dimensions = self.show_dimensions_act.isChecked()
        show_filesize = self.show_filesize_act.isChecked()

        self.progress_bar.setRange(0, total_files)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Loading {total_files} images...")

        for i, f in enumerate(files):
            full_p = os.path.join(path, f)

            # Determine display text (mutually exclusive)
            if show_names:
                display_text = f
            elif show_dimensions:
                display_text = self.get_image_dimensions_str(full_p)
            elif show_filesize:
                display_text = self.get_file_size_str(full_p)
            else:
                display_text = ""

            item = QListWidgetItem(QIcon(full_p), display_text)
            item.setData(Qt.ItemDataRole.UserRole, full_p)
            self.thumb_list.addItem(item)

            self.progress_bar.setValue(i + 1)

            if i % 10 == 0:
                QApplication.processEvents()

        self.progress_bar.setVisible(False)

        if self.filter_active and self.filter_mode:
            if self.filter_mode == "has_face":
                filter_name = "with faces"
            elif self.filter_mode == "no_face":
                filter_name = "without faces"
            else:
                filter_name = "matching AI filters"
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
        elif self.sort_by == "filesize":
            files = sorted(files, key=lambda f: self.get_file_size(os.path.join(path, f)),
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

    def get_file_size(self, file_path):
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0

    def remove_thumbnails_by_path(self, paths_to_remove):
        """Removes specific items from the thumb_list without a full reload."""
        paths_to_remove = {os.path.normpath(p) for p in paths_to_remove}

        # Find the lowest index of items being removed to select next item
        first_removed_index = None
        for i in range(self.thumb_list.count() - 1, -1, -1):
            item = self.thumb_list.item(i)
            item_path = os.path.normpath(item.data(Qt.ItemDataRole.UserRole))

            if item_path in paths_to_remove:
                if first_removed_index is None or i < first_removed_index:
                    first_removed_index = i
                self.thumb_list.takeItem(i)

        # Select the next thumbnail after the removed items
        if first_removed_index is not None and self.thumb_list.count() > 0:
            # Select the item at the same index (which is now the next item)
            # or the last item if we removed from the end
            next_index = min(first_removed_index, self.thumb_list.count() - 1)
            next_item = self.thumb_list.item(next_index)
            if next_item:
                self.thumb_list.setCurrentItem(next_item)
                self.thumb_list.scrollToItem(next_item)

        self.status_bar.showMessage(f"Removed {len(paths_to_remove)} items.", 3000)

        if self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

    def start_operation_batch(self, operation_type):
        """Start a new operation batch for undo tracking."""
        self.last_operation = {'type': operation_type, 'files': []}

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
                # Track copy operation for undo (only if operation batch is active)
                if self.last_operation and self.last_operation['type'] == 'copy':
                    self.last_operation['files'].append((src, dest))
            else:
                shutil.move(src, dest)
                # Track move operation for undo (only if operation batch is active)
                if self.last_operation and self.last_operation['type'] == 'move':
                    self.last_operation['files'].append((src, dest))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not transfer file: {e}")

    def remove_duplicates(self):
        """Remove duplicate images based on the selected deduplication mode."""
        import hashlib
        import subprocess

        # Get files to process
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            # Use selected files
            file_paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
            scope_msg = f"{len(file_paths)} selected file(s)"
        else:
            # Use all files in thumbnail list
            file_paths = [self.thumb_list.item(i).data(Qt.ItemDataRole.UserRole)
                         for i in range(self.thumb_list.count())]
            scope_msg = f"all {len(file_paths)} file(s) in directory"

        if len(file_paths) < 2:
            QMessageBox.information(self, "Remove Duplicates",
                                   "Need at least 2 images to find duplicates.")
            return

        # Confirm operation
        mode_descriptions = {
            "exact": "MD5 checksum comparison",
            "visual_largest": "visual similarity (retain largest)",
            "visual_smallest": "visual similarity (retain smallest)"
        }
        mode_desc = mode_descriptions.get(self.dedup_mode, self.dedup_mode)

        reply = QMessageBox.question(
            self, "Remove Duplicates",
            f"Find and remove duplicates from {scope_msg} using {mode_desc}?\n\n"
            "This will permanently delete duplicate files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Show progress
        self.status_bar.showMessage("Finding duplicates...", 0)
        QApplication.processEvents()

        try:
            if self.dedup_mode == "exact":
                duplicates = self.find_exact_duplicates(file_paths)
            else:
                duplicates = self.find_visual_duplicates(file_paths)

            if not duplicates:
                self.status_bar.showMessage("No duplicates found", 3000)
                QMessageBox.information(self, "Remove Duplicates", "No duplicate images found.")
                return

            # Delete duplicates
            deleted_count = self.delete_duplicate_files(duplicates)

            # Refresh thumbnail view
            self.refresh_thumbnails()

            self.status_bar.showMessage(f"Removed {deleted_count} duplicate file(s)", 3000)
            QMessageBox.information(self, "Remove Duplicates",
                                   f"Successfully removed {deleted_count} duplicate file(s).")

        except Exception as e:
            self.status_bar.showMessage("Error during deduplication", 3000)
            QMessageBox.critical(self, "Error", f"Failed to remove duplicates:\n{e}")

    def find_exact_duplicates(self, file_paths):
        """Find exact duplicates using MD5 checksums. Returns dict of {hash: [files]}."""
        import hashlib

        hash_dict = {}
        for path in file_paths:
            try:
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash not in hash_dict:
                        hash_dict[file_hash] = []
                    hash_dict[file_hash].append(path)
            except Exception as e:
                print(f"Error hashing {path}: {e}")

        # Return only groups with duplicates (more than 1 file)
        return {h: files for h, files in hash_dict.items() if len(files) > 1}

    def find_visual_duplicates(self, file_paths):
        """Find visual duplicates using findimagedupes. Returns dict of {group_id: [files]}."""
        import subprocess
        import tempfile
        import shutil

        # Check if findimagedupes is available
        try:
            subprocess.run(['findimagedupes', '--version'],
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            QMessageBox.warning(
                self, "Tool Not Found",
                "The 'findimagedupes' tool is not installed.\n\n"
                "Install it with:\n"
                "  Ubuntu/Debian: sudo apt install findimagedupes\n"
                "  Fedora: sudo dnf install findimagedupes\n"
                "  Arch: sudo pacman -S findimagedupes\n\n"
                "Or from source: https://github.com/jhnc/findimagedupes"
            )
            raise RuntimeError("findimagedupes not found")

        # Get unique directories containing the files
        directories = list(set(os.path.dirname(p) for p in file_paths))
        file_paths_set = set(file_paths)

        # Create a temporary directory for the fingerprint database
        temp_dir = tempfile.mkdtemp()
        db_file = os.path.join(temp_dir, "fingerprints.db")

        try:
            # Use custom hook to output files with ||| delimiter (handles spaces in paths)
            custom_hook = 'VIEW(){ for f in "$@"; do printf "%s|||" "$f"; done; echo; }'

            # Run findimagedupes on the directories
            # -f is for fingerprint database, -i is for custom view hook
            result = subprocess.run(
                ['findimagedupes', '-f', db_file, '-i', custom_hook] + directories,
                capture_output=True,
                text=True
            )

            # Parse output - each line has files separated by |||
            duplicates = {}
            group_id = 0

            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                # Split by ||| delimiter
                files = [f.strip() for f in line.strip().split('|||') if f.strip()]

                # Filter to only include files from our input set
                valid_files = [f for f in files if f in file_paths_set and os.path.isfile(f)]

                if len(valid_files) > 1:
                    duplicates[group_id] = valid_files
                    group_id += 1

            return duplicates

        finally:
            # Clean up temp directory and database
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def delete_duplicate_files(self, duplicates):
        """Delete duplicate files, keeping one per group based on dedup_mode.

        Args:
            duplicates: dict of {group_id: [file_paths]}

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        for group_id, files in duplicates.items():
            if len(files) < 2:
                continue

            # Determine which file to keep
            if self.dedup_mode == "exact":
                # For exact duplicates, keep the first one (arbitrary)
                keep_file = files[0]
                delete_files = files[1:]
            elif self.dedup_mode == "visual_largest":
                # Keep the largest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1], reverse=True)
                keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            elif self.dedup_mode == "visual_smallest":
                # Keep the smallest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1])
                keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            else:
                continue

            # Delete the duplicate files
            for file_path in delete_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        return deleted_count

    def remove_folder_duplicates(self, tree):
        """Remove duplicate images in selected folders based on folder dedup settings.

        Args:
            tree: The FolderTree widget (source or destination)
        """
        import hashlib
        import subprocess

        # Ensure folder dedup menu is set up
        if not hasattr(self, 'folder_dedup_menu'):
            self.setup_folder_dedup_menu()

        # Get selected folder paths
        selected_indexes = tree.selectedIndexes()
        # Filter to column 0 only to avoid duplicates from multiple columns
        selected_paths = []
        seen_rows = set()
        for idx in selected_indexes:
            if idx.column() == 0 and idx.row() not in seen_rows:
                seen_rows.add(idx.row())
                path = tree.model().filePath(idx)
                if os.path.isdir(path):
                    selected_paths.append(path)

        if not selected_paths:
            # Use root path if nothing selected
            if tree == self.src_tree:
                selected_paths = [self.src_root_path]
            else:
                selected_paths = [self.dest_root_path]

        if not selected_paths or not selected_paths[0]:
            QMessageBox.information(self, "Remove Duplicates",
                                   "Please select one or more folders first.")
            return

        # Collect all image files from selected folders
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif', '.tiff', '.bmp')
        file_paths = []

        for folder_path in selected_paths:
            if self.folder_dedup_recursive:
                # Recursive: walk all subdirectories
                for root, dirs, files in os.walk(folder_path):
                    for f in files:
                        if f.lower().endswith(valid_extensions):
                            file_paths.append(os.path.join(root, f))
            else:
                # Non-recursive: only immediate directory
                try:
                    for f in os.listdir(folder_path):
                        if f.lower().endswith(valid_extensions):
                            full_path = os.path.join(folder_path, f)
                            if os.path.isfile(full_path):
                                file_paths.append(full_path)
                except (PermissionError, OSError) as e:
                    print(f"Error accessing {folder_path}: {e}")

        if len(file_paths) < 2:
            QMessageBox.information(self, "Remove Duplicates",
                                   "Need at least 2 images to find duplicates.")
            return

        # Build scope message
        folder_names = [os.path.basename(p) for p in selected_paths]
        if len(folder_names) > 3:
            scope_msg = f"{len(folder_names)} folders"
        else:
            scope_msg = ", ".join(folder_names)
        if self.folder_dedup_recursive:
            scope_msg += " (recursive)"

        # Confirm operation
        mode_descriptions = {
            "exact": "MD5 checksum comparison",
            "visual_largest": "visual similarity (retain largest)",
            "visual_smallest": "visual similarity (retain smallest)",
            "visual_deepest": "visual similarity (retain deepest path)",
            "visual_shallow": "visual similarity (retain shallowest path)"
        }
        mode_desc = mode_descriptions.get(self.folder_dedup_mode, self.folder_dedup_mode)

        reply = QMessageBox.question(
            self, "Remove Duplicates",
            f"Find and remove duplicates from {scope_msg} ({len(file_paths)} files) using {mode_desc}?\n\n"
            "This will permanently delete duplicate files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Show progress
        self.status_bar.showMessage("Finding duplicates...", 0)
        QApplication.processEvents()

        try:
            if self.folder_dedup_mode == "exact":
                duplicates = self.find_exact_duplicates(file_paths)
            else:
                duplicates = self.find_visual_duplicates(file_paths)

            if not duplicates:
                self.status_bar.showMessage("No duplicates found", 3000)
                QMessageBox.information(self, "Remove Duplicates", "No duplicate images found.")
                return

            # Delete duplicates using folder-specific mode
            deleted_count = self.delete_folder_duplicate_files(duplicates)

            # Refresh thumbnail view if it shows affected folder
            self.refresh_thumbnails()

            # Update folder counts
            if self.show_counts_act.isChecked():
                self.src_tree.count_delegate._count_cache.clear()
                self.dest_tree.count_delegate._count_cache.clear()
                self.src_tree.viewport().update()
                self.dest_tree.viewport().update()

            self.status_bar.showMessage(f"Removed {deleted_count} duplicate file(s)", 3000)
            QMessageBox.information(self, "Remove Duplicates",
                                   f"Successfully removed {deleted_count} duplicate file(s).")

        except Exception as e:
            self.status_bar.showMessage("Error during deduplication", 3000)
            QMessageBox.critical(self, "Error", f"Failed to remove duplicates:\n{e}")

    def delete_folder_duplicate_files(self, duplicates):
        """Delete duplicate files using folder dedup mode.

        Args:
            duplicates: dict of {group_id: [file_paths]}

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        for group_id, files in duplicates.items():
            if len(files) < 2:
                continue

            # Determine which file to keep based on folder_dedup_mode
            if self.folder_dedup_mode == "exact":
                # For exact duplicates, keep the first one (arbitrary)
                keep_file = files[0]
                delete_files = files[1:]
            elif self.folder_dedup_mode == "visual_largest":
                # Keep the largest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1], reverse=True)
                keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            elif self.folder_dedup_mode == "visual_smallest":
                # Keep the smallest file
                files_with_size = [(f, os.path.getsize(f)) for f in files]
                files_with_size.sort(key=lambda x: x[1])
                keep_file = files_with_size[0][0]
                delete_files = [f for f, _ in files_with_size[1:]]
            elif self.folder_dedup_mode == "visual_deepest":
                # Keep the file with the deepest path (most directory separators)
                files_with_depth = [(f, f.count(os.sep)) for f in files]
                files_with_depth.sort(key=lambda x: x[1], reverse=True)
                keep_file = files_with_depth[0][0]
                delete_files = [f for f, _ in files_with_depth[1:]]
            elif self.folder_dedup_mode == "visual_shallow":
                # Keep the file with the shallowest path (fewest directory separators)
                files_with_depth = [(f, f.count(os.sep)) for f in files]
                files_with_depth.sort(key=lambda x: x[1])
                keep_file = files_with_depth[0][0]
                delete_files = [f for f, _ in files_with_depth[1:]]
            else:
                continue

            # Delete the duplicate files
            for file_path in delete_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        return deleted_count

    def undo_last_operation(self):
        """Undo the last move, copy, or delete operation."""
        # First check if there's a delete to undo (takes priority as it's the most recent)
        if hasattr(self, 'delete_undo_manager') and self.delete_undo_manager.can_undo():
            self.undo_delete()
            return

        if not self.last_operation:
            self.status_bar.showMessage("Nothing to undo", 3000)
            return

        operation_type = self.last_operation['type']
        files = self.last_operation['files']

        if not files:
            self.last_operation = None
            return

        # Save current scroll position
        scroll_bar = self.thumb_list.verticalScrollBar()
        saved_scroll_pos = scroll_bar.value()

        success_count = 0
        failed_count = 0

        for src, dest in files:
            try:
                if operation_type == "copy":
                    # For copy, delete the copied file
                    if os.path.exists(dest):
                        os.remove(dest)
                        success_count += 1
                    else:
                        failed_count += 1
                elif operation_type == "move":
                    # For move, move the file back to original location
                    if os.path.exists(dest):
                        shutil.move(dest, src)
                        success_count += 1
                    else:
                        failed_count += 1
            except Exception as e:
                failed_count += 1
                QMessageBox.warning(self, "Undo Error", f"Could not undo {os.path.basename(dest)}:\n{e}")

        # Clear the operation history after undo
        self.last_operation = None

        # Refresh thumbnails and folder counts
        self.refresh_thumbnails()

        # Restore scroll position after thumbnails are reloaded
        QTimer.singleShot(100, lambda: scroll_bar.setValue(saved_scroll_pos))

        # Update folder count caches
        if hasattr(self.src_tree, 'count_delegate') and hasattr(self.dest_tree, 'count_delegate'):
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
            self.src_tree.viewport().update()
            self.dest_tree.viewport().update()

        self.status_bar.showMessage(f"Undid {operation_type}: {success_count} file(s)", 3000)

    def undo_delete(self):
        """Undo the last file deletion."""
        if not hasattr(self, 'delete_undo_manager') or not self.delete_undo_manager.can_undo():
            self.status_bar.showMessage("Nothing to undo", 3000)
            return

        # Save current scroll position
        scroll_bar = self.thumb_list.verticalScrollBar()
        saved_scroll_pos = scroll_bar.value()

        # Restore the deleted files
        restored_paths = self.delete_undo_manager.undo()

        if restored_paths:
            # Refresh thumbnails to show restored files
            self.refresh_thumbnails()

            # Restore scroll position after thumbnails are reloaded
            QTimer.singleShot(100, lambda: scroll_bar.setValue(saved_scroll_pos))

            # Update folder count caches
            if hasattr(self.src_tree, 'count_delegate') and hasattr(self.dest_tree, 'count_delegate'):
                self.src_tree.count_delegate._count_cache.clear()
                self.dest_tree.count_delegate._count_cache.clear()
                self.src_tree.viewport().update()
                self.dest_tree.viewport().update()

            self.status_bar.showMessage(f"Restored {len(restored_paths)} file(s)", 3000)
        else:
            self.status_bar.showMessage("Could not restore deleted file(s)", 3000)

    def resizeEvent(self, event):
        """Handle main window resize - triggers thumbnail redraw."""
        super().resizeEvent(event)
        if hasattr(self, 'thumb_list') and hasattr(self, '_settings_loaded'):
            self.thumb_list.scheduleDelayedItemsLayout()

            if self._settings_loaded:
                self.write_settings()

    def open_yolo_window(self):
        """Open the YOLO labeling window with current thumbnails."""
        # Get the current image list from thumbnails
        image_list = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_list.append(path)

        if not image_list:
            QMessageBox.information(
                self, "No Images",
                "Please select a source folder with images first."
            )
            return

        # Get current selection index, default to 0
        current_index = 0
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            current_index = self.thumb_list.row(selected_items[0])

        # Open YOLO window - use showMaximized() for immediate full-screen
        self.yolo_window = YoloWindow(image_list, current_index, self)
        self.yolo_window.showMaximized()

    def open_caption_window(self):
        """Open the Auto Caption window with current thumbnails."""
        from caption_window import CaptionWindow

        # Get the current image list from thumbnails
        image_list = []
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                image_list.append(path)

        if not image_list:
            QMessageBox.information(
                self, "No Images",
                "Please select a source folder with images first."
            )
            return

        # Get current selection index, default to 0
        current_index = 0
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            current_index = self.thumb_list.row(selected_items[0])

        # Open Caption window
        self.caption_window = CaptionWindow(image_list, current_index, self)
        self.caption_window.showMaximized()

    def batch_generate_captions(self):
        """Generate captions for selected or all thumbnail images."""
        from caption_generator import get_caption_generator, CaptionWorker

        # Get images to process - selected items or all if none selected
        selected_items = self.thumb_list.selectedItems()
        if selected_items:
            image_paths = []
            for item in selected_items:
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_paths.append(path)
        else:
            # No selection - use all thumbnails
            image_paths = []
            for i in range(self.thumb_list.count()):
                item = self.thumb_list.item(i)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    image_paths.append(path)

        if not image_paths:
            QMessageBox.information(
                self, "No Images",
                "No images available to caption."
            )
            return

        # Filter out images that already have captions
        images_without_captions = []
        images_with_captions = []
        for path in image_paths:
            caption_path = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(caption_path):
                images_with_captions.append(path)
            else:
                images_without_captions.append(path)

        # Ask what to do if some already have captions
        if images_with_captions and images_without_captions:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Existing Captions")
            msg_box.setText(
                f"{len(images_with_captions)} image(s) already have captions.\n"
                f"{len(images_without_captions)} need captions.\n\n"
                "What would you like to do?"
            )
            skip_btn = msg_box.addButton("Skip existing", QMessageBox.ButtonRole.AcceptRole)
            overwrite_btn = msg_box.addButton("Overwrite all", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(skip_btn)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == skip_btn:
                # Skip existing - only process images without captions
                image_paths = images_without_captions
            # overwrite_btn = process all (overwrite existing)
        elif images_with_captions and not images_without_captions:
            # All have captions
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("All Have Captions")
            msg_box.setText(
                f"All {len(images_with_captions)} image(s) already have captions.\n"
                "Do you want to regenerate them?"
            )
            overwrite_btn = msg_box.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg_box.setDefaultButton(cancel_btn)
            msg_box.exec()

            if msg_box.clickedButton() != overwrite_btn:
                return

        if not image_paths:
            QMessageBox.information(self, "Nothing to Do", "No images need captions.")
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Generating captions...", "Cancel", 0, len(image_paths), self
        )
        progress.setWindowTitle("Batch Caption Generation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Get caption settings
        backend = getattr(self, 'caption_backend', 'florence2')
        ollama_url = getattr(self, 'caption_ollama_url', 'http://localhost:11434')
        ollama_model = getattr(self, 'caption_ollama_model', 'llava')
        florence_model = getattr(self, 'caption_florence_model', 'base')
        florence_detail = getattr(self, 'caption_florence_detail', 'detailed')
        wd14_model = getattr(self, 'caption_wd14_model', 'vit-v2')
        wd14_threshold = getattr(self, 'caption_wd14_threshold', 0.55)

        # Get generator instance
        generator = get_caption_generator()

        # Process images
        successful = 0
        failed = 0
        errors = []

        for i, image_path in enumerate(image_paths):
            if progress.wasCanceled():
                break

            # Update progress
            filename = os.path.basename(image_path)
            progress.setLabelText(f"Generating caption for:\n{filename}\n\n({i+1} of {len(image_paths)})")
            progress.setValue(i)
            QApplication.processEvents()

            try:
                # Generate caption
                if backend == "florence2":
                    caption = generator.generate_florence(
                        image_path,
                        model_key=florence_model,
                        detail_level=florence_detail
                    )
                elif backend == "wd14":
                    caption = generator.generate_wd14(
                        image_path,
                        model_key=wd14_model,
                        threshold=wd14_threshold
                    )
                else:  # ollama
                    caption = generator.generate_ollama(
                        image_path, ollama_url, ollama_model
                    )

                # Save caption to txt file
                caption_path = os.path.splitext(image_path)[0] + ".txt"
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                successful += 1

            except Exception as e:
                failed += 1
                errors.append(f"{filename}: {str(e)[:50]}")

            QApplication.processEvents()

        progress.setValue(len(image_paths))
        progress.close()

        # Show results
        if progress.wasCanceled():
            msg = f"Batch captioning cancelled.\n\nCompleted: {successful}\nFailed: {failed}"
        else:
            msg = f"Batch captioning complete.\n\nSuccessful: {successful}\nFailed: {failed}"

        if errors and len(errors) <= 5:
            msg += f"\n\nErrors:\n" + "\n".join(errors)
        elif errors:
            msg += f"\n\nFirst 5 errors:\n" + "\n".join(errors[:5])
            msg += f"\n... and {len(errors) - 5} more"

        QMessageBox.information(self, "Batch Caption Results", msg)

    def show_about(self):
        """Show the About dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("About Visual Image Master")
        dialog.setMinimumSize(450, 350)

        layout = QVBoxLayout(dialog)

        # Application title
        title_label = QLabel("<h1>Visual Image Master 2026</h1>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Version and description
        info_text = """
        <p style='text-align: center;'>
        <b>Version 1.0.0</b><br><br>
        A powerful desktop application for browsing, sorting,<br>
        filtering, and organizing image collections.<br><br>
        <b>Features:</b><br>
        Three-pane interface with drag-and-drop support<br>
        Face detection and AI-powered filtering<br>
        Visual duplicate detection and removal<br>
        Full-screen preview with crop functionality<br>
        Batch file operations with undo support<br><br>
        <b>Technologies:</b><br>
        Python 3 | PySide6 (Qt6) | OpenCV | Pillow<br><br>
        <hr>
        <small>
        Copyright 2026. All rights reserved.<br>
        Licensed under the MIT License.
        </small>
        </p>
        """
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # OK button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.exec()

    def show_manual(self):
        """Show the Manual/Help dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Visual Image Master - Manual")
        dialog.setMinimumSize(700, 600)

        layout = QVBoxLayout(dialog)

        # Create text browser for markdown-style content
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setReadOnly(True)

        manual_text = self.get_manual_text()
        text_browser.setMarkdown(manual_text)

        layout.addWidget(text_browser)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec()

    def get_manual_text(self):
        """Return the manual/help text content."""
        return """
# Visual Image Master 2026

A powerful desktop application for browsing, sorting, filtering, and organizing large image collections with an intuitive three-pane interface.

## Overview

Visual Image Master provides a streamlined workflow for managing images through:
- **Source Tree (Left)**: Browse and select source directories
- **Thumbnail Grid (Center)**: View and interact with images
- **Destination Tree (Right)**: Select target folders for organizing

## Getting Started

### Installation

```bash
pip install -r requirements.txt
python main.py
```

### Requirements
- Python 3.8+
- PySide6 (Qt6 bindings)
- Pillow (image processing)
- OpenCV (face detection and quality analysis)
- findimagedupes (optional, for visual duplicate detection)

## Interface Guide

### Navigation

- **Source/Destination Trees**: Click folders to navigate; use path entry box to type paths directly
- **Browse Button (...)**: Open folder selection dialog
- **Thumbnail Grid**: Click to select, Ctrl+click for multi-select, Shift+click for range select

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Delete` | Delete selected files (with undo) |
| `Shift+Delete` | Permanently delete (no undo) |
| `Ctrl+Z` | Undo last operation |
| `Ctrl+R` | Rotate image 90° clockwise |
| `Ctrl+S` | Save cropped image (in preview) |
| `Enter` | Open preview window |
| `Ctrl+Q` | Exit application |
| `F1` | Open this manual |
| `Ctrl+Wheel` | Zoom thumbnails/trees/preview |

### Mouse Controls

- **Double-click thumbnail**: Open full-screen preview
- **Drag thumbnails**: Move/copy to destination folder
- **Ctrl+Drag**: Copy instead of move
- **Right-click thumbnail**: Context menu with file operations
- **Right-click folder**: Create, rename, or delete folders

## Features

### Filtering Images

Click the filter button (🔽) in the toolbar:
- **Left-click**: Toggle filter on/off
- **Right-click**: Select filter mode

Filter modes:
- **Has Face**: Show only images with detected faces
- **No Face**: Show only images without faces
- **Advanced**: Configure AI-powered filters (objects, quality, scenes)

### Sorting Images

Click the sort button (⇅) to access sort options:
- Name (A-Z or Z-A)
- Date Modified
- Resolution
- Width / Height
- Quality Score
- File Size

### Duplicate Removal

**Thumbnail Pane** (🗑️ button):
- Works on selected images or all images in view
- Right-click for mode options

**Folder Trees** (🗑️ button next to path):
- Works on selected folders (multi-select with Ctrl/Shift)
- Supports recursive processing

Deduplication modes:
- **Exact (MD5)**: Remove byte-identical files
- **Visual - Retain Largest**: Keep highest resolution
- **Visual - Retain Smallest**: Keep smallest file size
- **Visual - Retain Deepest**: Keep file deepest in folder structure
- **Visual - Retain Shallow**: Keep file closest to root

### Preview Window

Double-click any thumbnail or press Enter to open the preview:
- **Arrow keys**: Navigate between images
- **Escape**: Close preview
- **Scroll wheel**: Zoom in/out (when not in fit mode)

**Crop Functionality** (View > Crop menu):
- Rectangle: Free-form resizable crop
- Fixed sizes: 1024×1024, 768×768, 640×640, 512×512
- Drag to reposition, drag corners to resize
- `Ctrl+S` to save cropped image

### File Operations

**Moving/Copying Files**:
1. Select thumbnails in the grid
2. Drag to destination folder, or
3. Right-click > Move/Copy to Selected, or
4. Right-click > Move/Copy to New (enter path)

**Undo Support**:
- `Ctrl+Z` undoes the last move, copy, or delete
- Delete operations store files temporarily for recovery
- Shift+Delete bypasses undo (permanent deletion)

## Options Menu

- **Remember Settings**: Persist window layout and preferences
- **Show Filenames**: Display names below thumbnails
- **Show Dimensions**: Display image dimensions
- **Show File Size**: Display file sizes
- **Show Folder Counts**: Show image counts in folder trees
- **Confirm Deletions**: Prompt before deleting files
- **Face Detection Settings**: Configure detection preset (Fast/Balanced/Thorough)

## Face Detection

Three detection presets:
- **Fast**: Haar Cascade only (quickest, less accurate)
- **Balanced**: DNN-based detection (recommended)
- **Thorough**: Both methods combined (most accurate)

For best results with DNN detection, download the model files:
```bash
cd models
python download_models.py
```

## Tips & Tricks

1. **Quick Navigation**: Type or paste paths directly in the path entry boxes
2. **Batch Operations**: Select multiple files with Ctrl+click before moving/copying
3. **Zoom Levels**: Use Ctrl+wheel to zoom thumbnails, folder trees, or preview
4. **Filter Workflow**: Enable face filter to quickly sort portrait vs. landscape photos
5. **Duplicate Cleanup**: Use recursive mode on parent folders to scan entire hierarchies

## Troubleshooting

**"findimagedupes not found"**: Install the tool for visual duplicate detection:
```bash
# Ubuntu/Debian
sudo apt install findimagedupes

# Fedora
sudo dnf install findimagedupes

# Arch
sudo pacman -S findimagedupes
```

**Slow face detection**: Switch to "Fast" preset in Options > Face Detection Settings

**Missing DNN models**: Run `python models/download_models.py` to download detection models

---

For bug reports and feature requests, visit the project repository.
"""

    def closeEvent(self, event):
        """Save settings and clean up when the application closes."""
        self.write_settings()
        # Clean up delete undo temp directory
        if hasattr(self, 'delete_undo_manager'):
            self.delete_undo_manager.cleanup()
        event.accept()
