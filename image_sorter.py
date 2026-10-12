"""
Main ImageSorter application class.
"""

import os
import shutil
import tempfile
import atexit
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QFileSystemModel,
    QToolButton,
    QMenu,
    QPushButton,
    QFileDialog,
    QLabel,
    QCheckBox,
)
from PySide6.QtCore import Qt, QSize, QDir, QTimer, QFileSystemWatcher
from PySide6.QtGui import QAction
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


from thumbnail_list import ThumbnailList  # noqa: E402
from folder_tree import FolderTree, FolderSortFilterProxyModel  # noqa: E402
from face_detection import FaceDetector, ImageQualityAnalyzer  # noqa: E402
from composition_similarity import CompositionAnalyzer  # noqa: E402
from face_detection_dialog import FaceDetectionDialog  # noqa: E402
from settings_manager import SettingsManager  # noqa: E402
from image_sorter_filters import FilteringMixin  # noqa: E402
from image_sorter_operations import FileOperationsMixin  # noqa: E402


class ImageSorter(QMainWindow, FilteringMixin, FileOperationsMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Image Master 2026")
        self.resize(1600, 900)

        # --- Persistence Placeholders ---
        self.preview_fit_default = False
        self.preview_autoname_extract = False
        self.preview_crop_align_mode = "image"
        self.preview_extract_save_dir = ""
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
        self.enhance_model_dir = ""  # Path to AI upscale model files
        self.enhance_save_mode = "overwrite"  # "overwrite" or "copy"
        self.enhance_copy_suffix = "_enhanced"  # Suffix for copy mode
        self.enhance_last_operation = "sharpen"  # Last selected operation
        self.enhance_last_params = {}  # Dict of {op_key: {param_key: value}}
        self._settings_loaded = False
        self.src_root_path = None
        self.dest_root_path = None

        # Sorting and filtering state
        self.sort_by = "name"
        self.sort_ascending = True
        self.similarity_reference_path = None  # reference image for "similarity" sort
        self.similarity_reference_paths = []  # reference images for "similarity_multi" sort
        self.similarity_negative_paths = []  # negative images for "similarity_discriminative" sort
        self.filter_active = False
        self.filter_mode = None  # None, "has_face", "no_face", "advanced"
        self.advanced_filter_settings = {}  # Settings from advanced filter dialog

        # Folder tree sort state (persisted per pane)
        self.src_tree_sort_mode = "name"
        self.src_tree_sort_ascending = True
        self.dest_tree_sort_mode = "name"
        self.dest_tree_sort_ascending = True

        # Face detection preset
        self.face_detection_preset = "balanced"

        # Face detector (will be initialized with preset after settings load)
        self.face_detector = None

        # Image quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer()

        # Composition (colour-independent structural) similarity analyzer
        self.composition_analyzer = CompositionAnalyzer()

        # Advanced AI detectors (lazy-loaded)
        self._object_detector = None
        self._quality_assessor = None
        self._scene_recognizer = None

        # Undo state for move/copy operations
        self.last_operation = (
            None  # {"type": "move"|"copy", "files": [(src, dest), ...]}
        )

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
        file_menu = menu_bar.addMenu("&File")

        refresh_act = QAction("&Refresh Directory Tree", self)
        refresh_act.setShortcut("F5")
        refresh_act.setToolTip(
            "Re-scan both folder trees from disk - use this if a folder "
            "renamed/added/removed outside the app (or via the in-app "
            "Rename) isn't showing up correctly"
        )
        refresh_act.triggered.connect(self.refresh_directory_trees)
        file_menu.addAction(refresh_act)

        file_menu.addSeparator()

        exit_act = QAction("&Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Options Menu
        options_menu = menu_bar.addMenu("&Options")

        self.remember_settings_act = QAction("Remember Settings", self, checkable=True)
        self.remember_settings_act.setChecked(True)
        options_menu.addAction(self.remember_settings_act)

        self.show_names_act = QAction("Show Filenames", self, checkable=True)
        self.show_names_act.setChecked(False)
        self.show_names_act.triggered.connect(self.toggle_filenames)
        options_menu.addAction(self.show_names_act)

        self.show_dimensions_act = QAction("Show Dimensions", self, checkable=True)
        self.show_dimensions_act.setChecked(False)
        self.show_dimensions_act.triggered.connect(self.toggle_dimensions)
        options_menu.addAction(self.show_dimensions_act)

        self.show_filesize_act = QAction("Show File Size", self, checkable=True)
        self.show_filesize_act.setChecked(False)
        self.show_filesize_act.triggered.connect(self.toggle_filesize)
        options_menu.addAction(self.show_filesize_act)

        self.show_counts_act = QAction("Show Folder Counts", self, checkable=True)
        self.show_counts_act.setChecked(False)
        self.show_counts_act.triggered.connect(self.toggle_folder_counts)
        options_menu.addAction(self.show_counts_act)

        self.conf_delete_act = QAction("Confirm Deletions", self, checkable=True)
        self.conf_delete_act.setChecked(True)
        self.conf_delete_act.triggered.connect(self.toggle_conf_del)
        options_menu.addAction(self.conf_delete_act)

        self.comfyui_info_act = QAction("Show ComfyUI Info", self, checkable=True)
        self.comfyui_info_act.setChecked(False)
        self.comfyui_info_act.setToolTip(
            "Read LoRAs and prompt text embedded in ComfyUI-generated PNGs, "
            "shown in the preview window"
        )
        self.comfyui_info_act.triggered.connect(self.toggle_comfyui_info)
        options_menu.addAction(self.comfyui_info_act)

        options_menu.addSeparator()

        face_detect_settings_act = QAction("Face Detection Settings...", self)
        face_detect_settings_act.triggered.connect(self.show_face_detection_settings)
        options_menu.addAction(face_detect_settings_act)

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")

        yolo_labeling_act = QAction("&YOLO Labelling...", self)
        yolo_labeling_act.triggered.connect(self.open_yolo_window)
        tools_menu.addAction(yolo_labeling_act)

        slideshow_act = QAction("&Slideshow...", self)
        slideshow_act.triggered.connect(self.open_slideshow)
        tools_menu.addAction(slideshow_act)

        reference_consistency_act = QAction(
            "&Reference Folder Consistency Check...", self
        )
        reference_consistency_act.setToolTip(
            "Audit a reference-image folder for photos that don't facially "
            "match the rest of it (e.g. mislabeled/wrong-identity) before "
            "using it with the reference-folder similarity sorts"
        )
        reference_consistency_act.triggered.connect(
            self.open_reference_consistency_window
        )
        tools_menu.addAction(reference_consistency_act)

        auto_caption_act = QAction("&Auto Caption...", self)
        auto_caption_act.triggered.connect(self.open_caption_window)
        tools_menu.addAction(auto_caption_act)

        enhance_act = QAction("Image &Enhance...", self)
        enhance_act.triggered.connect(self.open_enhance_dialog)
        tools_menu.addAction(enhance_act)

        rotate_act = QAction("&Rotate Selected 90° Clockwise", self)
        rotate_act.setShortcut("Ctrl+R")
        rotate_act.triggered.connect(lambda: self.thumb_list.rotate_selected_images())
        tools_menu.addAction(rotate_act)

        pair_act = QAction("&Pair Selector...", self)
        pair_act.triggered.connect(self.open_pair_selector)
        tools_menu.addAction(pair_act)

        letterbox_act = QAction("Remove &Letterbox...", self)
        letterbox_act.triggered.connect(self.open_letterbox_dialog)
        tools_menu.addAction(letterbox_act)

        filename_matcher_act = QAction("&Filename Matcher...", self)
        filename_matcher_act.triggered.connect(self.open_filename_matcher)
        tools_menu.addAction(filename_matcher_act)

        re_extract_matcher_act = QAction("&Re-Extract Matcher...", self)
        re_extract_matcher_act.triggered.connect(self.open_re_extract_matcher)
        tools_menu.addAction(re_extract_matcher_act)

        analyze_ratings_act = QAction("&Analyze Ratings...", self)
        analyze_ratings_act.setToolTip(
            "Correlate rated images' ComfyUI LoRAs/strengths and prompt "
            "keywords with their ratings"
        )
        analyze_ratings_act.triggered.connect(self.open_rating_analysis)
        tools_menu.addAction(analyze_ratings_act)

        export_ratings_csv_act = QAction("Export Ratings &CSV...", self)
        export_ratings_csv_act.setToolTip(
            "Export rated images' ComfyUI LoRAs/strengths/prompts as CSV "
            "(one row per image-LoRA pair) for analysis in Excel/pandas/etc."
        )
        export_ratings_csv_act.triggered.connect(self.export_ratings_csv)
        tools_menu.addAction(export_ratings_csv_act)

        self.preview_rating_display_act = QAction(
            "Show Rating in Preview", self, checkable=True
        )
        self.preview_rating_display_act.setChecked(True)
        self.preview_rating_display_act.setToolTip(
            "Show the star rating for the current image in the preview "
            "window's status bar"
        )
        self.preview_rating_display_act.triggered.connect(
            self.toggle_preview_rating_display
        )
        tools_menu.addAction(self.preview_rating_display_act)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")

        manual_act = QAction("&Manual", self)
        manual_act.setShortcut("F1")
        manual_act.triggered.connect(self.show_manual)
        help_menu.addAction(manual_act)

        help_menu.addSeparator()

        about_act = QAction("&About", self)
        about_act.triggered.connect(self.show_about)
        help_menu.addAction(about_act)

        # --- Status Bar and Progress Bar Setup ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Simple text spinner shown alongside status messages during long operations
        self._spinner_chars = "|/-\\"
        self._spinner_index = 0
        self._spinner_base_message = ""
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(120)
        self._spinner_timer.timeout.connect(self._advance_spinner)

        # Watches the currently displayed source folder so files added by an
        # external app (e.g. an export from another tool) appear in the
        # thumbnail grid without requiring a full manual reload.
        self.source_watcher = QFileSystemWatcher(self)
        self.source_watcher.directoryChanged.connect(self.on_source_directory_changed)
        self._pending_new_files = {}  # path -> last-seen size, for new-file debounce
        self._new_file_check_timer = QTimer(self)
        self._new_file_check_timer.setSingleShot(True)
        self._new_file_check_timer.timeout.connect(self._check_for_new_thumbnails)

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
        self.src_dedup_btn.setToolTip(
            "Remove duplicates in selected folders (right-click for options)"
        )
        self.src_dedup_btn.clicked.connect(
            lambda: self.remove_folder_duplicates(self.src_tree)
        )
        self.src_dedup_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.src_dedup_btn.customContextMenuRequested.connect(
            self.show_folder_dedup_menu
        )
        self.src_path_row.addWidget(self.src_path_display)
        self.src_path_row.addWidget(self.src_browse_btn)
        self.src_path_row.addWidget(self.src_dedup_btn)

        self.src_model = QFileSystemModel()
        self.src_model.setRootPath(QDir.rootPath())
        self.src_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.src_proxy = FolderSortFilterProxyModel()
        self.src_proxy.setSourceModel(self.src_model)
        self.src_tree = FolderTree(self, is_source=True)
        self.src_tree.set_model_and_clean_columns(self.src_proxy)
        self.src_proxy.filter_applied.connect(self._reassert_src_tree_root)

        # Source filter + sort row
        self.src_filter_row = QHBoxLayout()
        self.src_filter_row.setContentsMargins(0, 2, 0, 0)
        self.src_filter_edit = QLineEdit()
        self.src_filter_edit.setPlaceholderText("Filter folders...")
        self.src_filter_edit.textChanged.connect(
            lambda t: self.src_proxy.set_filter_text(t)
        )
        self.src_sort_btn = QToolButton()
        self.src_sort_btn.setText("⇅")
        self.src_sort_btn.setToolTip("Sort folder tree")
        self.src_sort_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.src_sort_menu = QMenu(self.src_sort_btn)
        self.src_tree_sort_actions = self._setup_tree_sort_menu(
            self.src_sort_menu, "src"
        )
        self.src_sort_btn.setMenu(self.src_sort_menu)
        self.src_regex_check = QCheckBox("Regex")
        self.src_regex_check.setToolTip(
            "Treat filter text as a regular expression instead of plain substring"
        )
        self.src_regex_check.toggled.connect(self.src_proxy.set_regex_mode)
        self.src_expand_btn = QToolButton()
        self.src_expand_btn.setCheckable(True)
        self.src_expand_btn.setText("+")
        self.src_expand_btn.setToolTip("Expand all matching directories")
        self.src_expand_btn.toggled.connect(
            lambda checked: self._on_expand_toggled(
                self.src_tree, self.src_expand_btn, checked
            )
        )
        self.src_filter_row.addWidget(self.src_filter_edit)
        self.src_filter_row.addWidget(self.src_regex_check)
        self.src_filter_row.addWidget(self.src_expand_btn)
        self.src_filter_row.addWidget(self.src_sort_btn)

        self.src_layout.addLayout(self.src_path_row)
        self.src_layout.addLayout(self.src_filter_row)
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
        self.filter_btn.setToolTip(
            "Left-click to toggle filter, right-click for options"
        )
        self.filter_btn.clicked.connect(self.toggle_filter)
        self.filter_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filter_btn.customContextMenuRequested.connect(self.show_filter_menu)
        self.filter_menu = QMenu(self.filter_btn)
        self.setup_filter_menu()

        # Store original button stylesheet
        self._filter_btn_original_style = (
            self.filter_btn.styleSheet() if self.filter_btn.styleSheet() else ""
        )

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
        self.batch_caption_btn.setToolTip(
            "Generate captions (left-click to run, right-click for settings)"
        )
        self.batch_caption_btn.clicked.connect(self.batch_generate_captions)
        self.batch_caption_btn.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.batch_caption_btn.customContextMenuRequested.connect(
            self.show_caption_settings_menu
        )

        # Selection count label
        self.selection_count_label = QLabel("Selected: 0")
        self.selection_count_label.setContentsMargins(8, 0, 0, 0)

        self.toolbar_layout.addWidget(self.filter_btn)
        self.toolbar_layout.addWidget(self.sort_btn)
        self.toolbar_layout.addWidget(self.dedup_btn)
        self.toolbar_layout.addWidget(self.batch_caption_btn)
        self.toolbar_layout.addWidget(self.selection_count_label)
        self.toolbar_layout.addStretch()

        self.thumb_list = ThumbnailList(self)
        self.thumb_list.itemSelectionChanged.connect(self.update_selection_count)
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
        self.dest_dedup_btn.setToolTip(
            "Remove duplicates in selected folders (right-click for options)"
        )
        self.dest_dedup_btn.clicked.connect(
            lambda: self.remove_folder_duplicates(self.dest_tree)
        )
        self.dest_dedup_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dest_dedup_btn.customContextMenuRequested.connect(
            self.show_folder_dedup_menu
        )
        self.dest_path_row.addWidget(self.dest_path_display)
        self.dest_path_row.addWidget(self.dest_browse_btn)
        self.dest_path_row.addWidget(self.dest_dedup_btn)

        self.dest_model = QFileSystemModel()
        self.dest_model.setRootPath(QDir.rootPath())
        self.dest_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.dest_proxy = FolderSortFilterProxyModel()
        self.dest_proxy.setSourceModel(self.dest_model)
        self.dest_tree = FolderTree(self, is_source=False)
        self.dest_tree.set_model_and_clean_columns(self.dest_proxy)
        self.dest_proxy.filter_applied.connect(self._reassert_dest_tree_root)

        # Destination filter + sort row
        self.dest_filter_row = QHBoxLayout()
        self.dest_filter_row.setContentsMargins(0, 2, 0, 0)
        self.dest_filter_edit = QLineEdit()
        self.dest_filter_edit.setPlaceholderText("Filter folders...")
        self.dest_filter_edit.textChanged.connect(
            lambda t: self.dest_proxy.set_filter_text(t)
        )
        self.dest_sort_btn = QToolButton()
        self.dest_sort_btn.setText("⇅")
        self.dest_sort_btn.setToolTip("Sort folder tree")
        self.dest_sort_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.dest_sort_menu = QMenu(self.dest_sort_btn)
        self.dest_tree_sort_actions = self._setup_tree_sort_menu(
            self.dest_sort_menu, "dest"
        )
        self.dest_sort_btn.setMenu(self.dest_sort_menu)
        self.dest_regex_check = QCheckBox("Regex")
        self.dest_regex_check.setToolTip(
            "Treat filter text as a regular expression instead of plain substring"
        )
        self.dest_regex_check.toggled.connect(self.dest_proxy.set_regex_mode)
        self.dest_expand_btn = QToolButton()
        self.dest_expand_btn.setCheckable(True)
        self.dest_expand_btn.setText("+")
        self.dest_expand_btn.setToolTip("Expand all matching directories")
        self.dest_expand_btn.toggled.connect(
            lambda checked: self._on_expand_toggled(
                self.dest_tree, self.dest_expand_btn, checked
            )
        )
        self.dest_filter_row.addWidget(self.dest_filter_edit)
        self.dest_filter_row.addWidget(self.dest_regex_check)
        self.dest_filter_row.addWidget(self.dest_expand_btn)
        self.dest_filter_row.addWidget(self.dest_sort_btn)

        self.dest_layout.addLayout(self.dest_path_row)
        self.dest_layout.addLayout(self.dest_filter_row)
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

    def start_loading_spinner(self, message):
        """Start showing an animated text spinner alongside a status message."""
        self._spinner_base_message = message
        self._spinner_index = 0
        self._advance_spinner()
        self._spinner_timer.start()

    def set_loading_spinner_message(self, message):
        """Change the base message shown next to the running spinner."""
        self._spinner_base_message = message

    def _advance_spinner(self):
        char = self._spinner_chars[self._spinner_index % len(self._spinner_chars)]
        self._spinner_index += 1
        self.status_bar.showMessage(f"{self._spinner_base_message} {char}")

    def stop_loading_spinner(self):
        """Stop the animated spinner. Caller is responsible for the final status message."""
        self._spinner_timer.stop()

    def update_selection_count(self):
        """Update the thumbnail selection count label shown above the grid."""
        count = len(self.thumb_list.selectedItems())
        self.selection_count_label.setText(f"Selected: {count}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for main window."""
        if (
            event.key() == Qt.Key.Key_Z
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.undo_last_operation()
        elif event.key() == Qt.Key.Key_Escape and getattr(
            self, "_refresh_in_progress", False
        ):
            self.cancel_sort_calculation()
        else:
            super().keyPressEvent(event)

    def _setup_tree_sort_menu(self, menu, side):
        """Build the sort popup menu for a folder tree pane. Returns {(mode,asc): action}."""
        options = [
            ("Name (A→Z)", "name", True),
            ("Name (Z→A)", "name", False),
            ("Date Modified (Newest first)", "date_modified", False),
            ("Date Modified (Oldest first)", "date_modified", True),
        ]
        sort_actions = {}
        for label, mode, asc in options:
            act = QAction(label, menu, checkable=True)
            act.triggered.connect(
                lambda checked, m=mode, a=asc, s=side: self._set_tree_sort(s, m, a)
            )
            menu.addAction(act)
            sort_actions[(mode, asc)] = act
        sort_actions[("name", True)].setChecked(True)
        return sort_actions

    def _set_tree_sort(self, side, mode, ascending):
        """Apply a sort mode to a folder tree pane and update menu check state."""
        if side == "src":
            self.src_proxy.set_sort_mode(mode, ascending)
            self.src_tree_sort_mode = mode
            self.src_tree_sort_ascending = ascending
            actions = self.src_tree_sort_actions
        else:
            self.dest_proxy.set_sort_mode(mode, ascending)
            self.dest_tree_sort_mode = mode
            self.dest_tree_sort_ascending = ascending
            actions = self.dest_tree_sort_actions
        for (m, a), act in actions.items():
            act.setChecked(m == mode and a == ascending)
        if self._settings_loaded:
            self.write_settings()

    def write_settings(self):
        """Delegate to settings manager."""
        self.settings_manager.write_settings()

    def read_settings(self):
        """Delegate to settings manager."""
        self.settings_manager.read_settings()

    def get_current_source_path(self):
        """Get the currently selected path in the source tree for loading thumbnails."""
        current_index = self.src_tree.currentIndex()
        if current_index.isValid():
            return self.src_model.filePath(self.src_proxy.mapToSource(current_index))
        return self.src_root_path if self.src_root_path else "/"

    def on_source_folder_changed(self, index):
        """Called when user selects a different folder in the source tree."""
        self.refresh_thumbnails()
        if self._settings_loaded:
            self.write_settings()

    def _on_expand_toggled(self, tree, button, checked):
        """Expand/collapse all directories currently visible in tree.

        Shared by both the source and destination panes' +/- toggle button.
        """
        if checked:
            tree.expandAll()
            button.setText("-")
            button.setToolTip("Collapse directory tree")
        else:
            tree.collapseAll()
            button.setText("+")
            button.setToolTip("Expand all matching directories")

    def _reassert_src_tree_root(self):
        """Re-derive the source tree's root index from src_root_path.

        A previously-captured proxy QModelIndex can go stale when the filter
        proxy rebuilds its row mapping (e.g. after invalidateFilter()),
        silently resetting the QTreeView to show the whole filesystem.
        """
        if not self.src_root_path:
            return
        proxy_index = self.src_proxy.mapFromSource(
            self.src_model.index(self.src_root_path)
        )
        if proxy_index.isValid():
            self.src_tree.setRootIndex(proxy_index)

    def _reassert_dest_tree_root(self):
        """Re-derive the destination tree's root index from dest_root_path.

        See _reassert_src_tree_root for why this is necessary.
        """
        if not self.dest_root_path:
            return
        proxy_index = self.dest_proxy.mapFromSource(
            self.dest_model.index(self.dest_root_path)
        )
        if proxy_index.isValid():
            self.dest_tree.setRootIndex(proxy_index)

    def refresh_directory_trees(self, refresh_thumbnails_too=True):
        """Force both folder trees to re-scan the filesystem.

        QFileSystemModel is normally self-updating via its own file-system
        watcher, but that isn't reliable for a directory that was renamed
        outside the model's own rename path (e.g. FolderTree.rename_prompt,
        which calls plain os.rename()) - the model's cached row for the old
        name can be left showing indefinitely. QFileSystemModel has no
        public "rescan" method; re-applying its current filter is the one
        operation found (empirically - setRootPath() toggling does *not*
        reliably do this) to force it to re-read an already-cached
        directory. Also available from File > Refresh Directory Tree (F5)
        as a manual escape hatch for any other external change.

        refresh_thumbnails_too is skipped by FolderTree.create_folder_prompt:
        a brand new (empty) subdirectory can never change which image files
        are showing in the currently browsed folder, so there's no reason
        to pay for a full re-sort/reload of the thumbnail grid just because
        a sibling or child folder appeared in the tree.
        """
        self.src_model.setFilter(self.src_model.filter())
        self.dest_model.setFilter(self.dest_model.filter())

        if self.show_counts_act.isChecked():
            self.src_tree.count_delegate._count_cache.clear()
            self.dest_tree.count_delegate._count_cache.clear()
        self.src_tree.viewport().update()
        self.dest_tree.viewport().update()

        self._reassert_src_tree_root()
        self._reassert_dest_tree_root()

        if refresh_thumbnails_too:
            self.refresh_thumbnails()
        self.status_bar.showMessage("Directory tree refreshed", 3000)

    def on_source_path_change(self):
        """Called when user manually edits the source path and presses Enter."""
        new_path = self.src_path_display.text()

        if not os.path.isdir(new_path):
            QMessageBox.warning(
                self,
                "Invalid Path",
                f"The path '{new_path}' does not exist or is not a directory.",
            )
            if self.src_root_path and os.path.isdir(self.src_root_path):
                self.src_path_display.setText(self.src_root_path)
            return

        new_path = os.path.abspath(new_path)
        self.src_path_display.setText(new_path)

        self.src_tree.blockSignals(True)
        self.src_root_path = new_path
        self.src_proxy.set_scope_root(new_path)
        proxy_index = self.src_proxy.mapFromSource(self.src_model.index(new_path))
        self.src_tree.setRootIndex(proxy_index)
        if proxy_index.isValid():
            self.src_tree.setCurrentIndex(proxy_index)
            self.src_tree.scrollTo(proxy_index)

        self.src_tree.blockSignals(False)
        self.refresh_thumbnails()

        if self._settings_loaded:
            self.write_settings()

    def on_dest_path_change(self):
        """Called when user manually edits the destination path and presses Enter."""
        new_path = self.dest_path_display.text()

        if not os.path.isdir(new_path):
            QMessageBox.warning(
                self,
                "Invalid Path",
                f"The path '{new_path}' does not exist or is not a directory.",
            )
            if self.dest_root_path and os.path.isdir(self.dest_root_path):
                self.dest_path_display.setText(self.dest_root_path)
            return

        new_path = os.path.abspath(new_path)
        self.dest_path_display.setText(new_path)

        self.dest_tree.blockSignals(True)
        self.dest_root_path = new_path
        self.dest_proxy.set_scope_root(new_path)
        proxy_index = self.dest_proxy.mapFromSource(self.dest_model.index(new_path))
        self.dest_tree.setRootIndex(proxy_index)
        if proxy_index.isValid():
            self.dest_tree.setCurrentIndex(proxy_index)
            self.dest_tree.scrollTo(proxy_index)

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
            self.src_proxy.set_scope_root(folder)
            proxy_index = self.src_proxy.mapFromSource(self.src_model.index(folder))
            self.src_tree.setRootIndex(proxy_index)
            if proxy_index.isValid():
                self.src_tree.setCurrentIndex(proxy_index)
                self.src_tree.scrollTo(proxy_index)
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
            self.dest_proxy.set_scope_root(folder)
            proxy_index = self.dest_proxy.mapFromSource(self.dest_model.index(folder))
            self.dest_tree.setRootIndex(proxy_index)
            if proxy_index.isValid():
                self.dest_tree.setCurrentIndex(proxy_index)
                self.dest_tree.scrollTo(proxy_index)
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

    def toggle_comfyui_info(self):
        state = self.comfyui_info_act.isChecked()
        status = "enabled" if state else "disabled"
        self.status_bar.showMessage(f"ComfyUI info {status}.", 3000)
        if self._settings_loaded:
            self.write_settings()

    def toggle_preview_rating_display(self):
        state = self.preview_rating_display_act.isChecked()
        status = "shown" if state else "hidden"
        self.status_bar.showMessage(f"Preview rating display {status}.", 3000)

        from preview_window import ImagePreviewWindow

        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, ImagePreviewWindow):
                widget._update_rating_label(
                    widget.image_list[widget.current_index]
                    if widget.image_list
                    and widget.current_index < len(widget.image_list)
                    else None
                )

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
                    f"Face detection preset changed to: {new_preset.capitalize()}", 5000
                )

                if self._settings_loaded:
                    self.write_settings()

    def resizeEvent(self, event):
        """Handle main window resize - triggers thumbnail redraw."""
        super().resizeEvent(event)
        if hasattr(self, "thumb_list") and hasattr(self, "_settings_loaded"):
            self.thumb_list.scheduleDelayedItemsLayout()

            if self._settings_loaded:
                self.write_settings()

    def closeEvent(self, event):
        """Save settings and clean up when the application closes."""
        self.write_settings()
        # Clean up delete undo temp directory
        if hasattr(self, "delete_undo_manager"):
            self.delete_undo_manager.cleanup()
        event.accept()
