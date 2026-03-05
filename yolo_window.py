"""
YOLO labeling window for creating training datasets.
"""

import os
from PySide6.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QLabel,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStatusBar,
    QApplication,
    QComboBox,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QInputDialog,
    QMessageBox,
    QMenu,
)
from PySide6.QtCore import Qt, QTimer, QSize, QPoint, QRect
from PySide6.QtGui import QAction, QPixmap, QKeySequence

from utils import find_main_window
from yolo_overlay import YoloOverlay
from yolo_label_panel import YoloLabelPanel, DEFAULT_COLORS
from yolo_file_handler import YoloFileHandler


# Available resolution presets
RESOLUTION_PRESETS = [
    ("Please select...", None),
    ("512 × 512", QSize(512, 512)),
    ("640 × 640", QSize(640, 640)),
    ("768 × 768", QSize(768, 768)),
    ("1024 × 1024", QSize(1024, 1024)),
]


class YoloWindow(QMainWindow):
    """Window for YOLO dataset labeling with image display and annotation tools."""

    # Size mode constants
    SIZE_MODE_IMAGE = "image"  # Selection size scales with image zoom
    SIZE_MODE_SCREEN = "screen"  # Selection size stays fixed on screen

    def __init__(self, image_list, current_index, parent=None):
        super().__init__(parent)

        # Establish the Main Window reference
        self.parent_app = parent
        if not self.parent_app:
            self.parent_app = find_main_window()

        # Core data
        self.image_list = image_list
        self.current_index = current_index
        self.zoom_factor = 1.0
        self.original_pixmap = QPixmap()

        # View settings
        self.fit_to_window = False

        # Selection box settings
        self.selection_size = None  # QSize in original image pixels (for image mode) or screen pixels (for screen mode)
        self.size_mode = self.SIZE_MODE_IMAGE  # Default to image resolution mode
        self.selection_locked = False  # Whether the selection size is locked

        # Track last used selection for restoring on new images
        self._last_selection_size = None  # QSize of last used selection
        self._last_size_mode = self.SIZE_MODE_IMAGE  # Mode of last used selection

        # Overlay widget
        self.overlay = None

        # Unsaved changes tracking
        self.has_unsaved_changes = False

        # Initialization flag - prevents re-initialization on subsequent show events
        self._initialized = False
        self._maximize_attempts = 0  # Counter for maximize retries

        # Set window title
        self.setWindowTitle("YOLO Labeling Tool")

        # Setup UI
        self._setup_ui()
        self._setup_menu()

        # Note: Window will be maximized in showEvent() after geometry is available

        # Event configuration
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scroll_area.installEventFilter(self)

    def _setup_ui(self):
        """Setup the main UI with splitter layout."""
        # Main splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: Image display area
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.splitter.addWidget(self.scroll_area)

        # Right side: Control panel
        self.right_panel = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        self.right_panel_layout.setContentsMargins(10, 10, 10, 10)

        # Resolution selection group
        resolution_group = QGroupBox("Selection Size")
        resolution_layout = QVBoxLayout(resolution_group)

        # Resolution dropdown
        self.resolution_combo = QComboBox()
        for label, size in RESOLUTION_PRESETS:
            self.resolution_combo.addItem(label, size)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        resolution_layout.addWidget(self.resolution_combo)

        # Size mode radio buttons
        mode_layout = QHBoxLayout()
        self.size_mode_group = QButtonGroup(self)

        self.image_mode_radio = QRadioButton("Image")
        self.image_mode_radio.setToolTip(
            "Selection size is in image pixels (scales with zoom)"
        )
        self.image_mode_radio.setChecked(True)
        self.size_mode_group.addButton(self.image_mode_radio)
        mode_layout.addWidget(self.image_mode_radio)

        self.screen_mode_radio = QRadioButton("Screen")
        self.screen_mode_radio.setToolTip(
            "Selection size is in screen pixels (fixed on screen)"
        )
        self.size_mode_group.addButton(self.screen_mode_radio)
        mode_layout.addWidget(self.screen_mode_radio)

        self.size_mode_group.buttonClicked.connect(self._on_size_mode_changed)
        resolution_layout.addLayout(mode_layout)

        # Set button row
        set_layout = QHBoxLayout()
        self.set_btn = QPushButton("Set")
        self.set_btn.setToolTip("Lock the current selection size for labeling")
        self.set_btn.clicked.connect(self._on_set_clicked)
        self.set_btn.setEnabled(False)  # Disabled until a resolution is selected
        set_layout.addWidget(self.set_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Clear the selection and unlock")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        self.clear_btn.setEnabled(False)
        set_layout.addWidget(self.clear_btn)

        resolution_layout.addLayout(set_layout)

        self.right_panel_layout.addWidget(resolution_group)

        # Extracted Images section
        self.extractions_group = QGroupBox("Extracted Images (0)")
        extractions_layout = QVBoxLayout(self.extractions_group)

        # Dropdown for existing extractions
        self.extractions_combo = QComboBox()
        self.extractions_combo.setToolTip(
            "Select a previously extracted image to load its annotations"
        )
        self.extractions_combo.currentIndexChanged.connect(self._on_extraction_selected)
        self.extractions_combo.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.extractions_combo.customContextMenuRequested.connect(
            self._show_extractions_context_menu
        )
        extractions_layout.addWidget(self.extractions_combo)

        self.right_panel_layout.addWidget(self.extractions_group)

        # Label panel (labels list, output path, save button)
        self.label_panel = YoloLabelPanel()
        self.label_panel.label_selected.connect(self._on_label_selected)
        self.label_panel.label_deleted.connect(self._on_label_deleted)
        self.label_panel.labels_renumbered.connect(self._on_labels_renumbered)
        self.label_panel.labels_changed.connect(self._on_labels_changed)
        self.label_panel.output_path_changed.connect(self._on_output_path_changed)
        self.label_panel.save_requested.connect(self._on_save_requested)
        self.label_panel.load_directory_requested.connect(
            self._on_load_directory_requested
        )
        self.label_panel.refresh_files_requested.connect(
            self._on_refresh_files_requested
        )
        self.right_panel_layout.addWidget(self.label_panel, 1)  # Stretch to fill

        self.splitter.addWidget(self.right_panel)

        # Set splitter proportions (75% image, 25% panel)
        self.splitter.setSizes([750, 250])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Selection size display
        self.selection_label = QLabel()
        self.selection_label.setStyleSheet("QLabel { padding: 2px; }")
        self.status_bar.addWidget(self.selection_label)

        # Zoom level display
        self.zoom_label = QLabel()
        self.zoom_label.setStyleSheet("QLabel { padding: 2px; margin-left: 10px; }")
        self.status_bar.addPermanentWidget(self.zoom_label)

    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # Options menu
        options_menu = menubar.addMenu("Options")

        # Fit to Window Action
        self.fit_act = QAction("Fit to Window", self, checkable=True)
        self.fit_act.setChecked(self.fit_to_window)
        self.fit_act.triggered.connect(self.toggle_fit_mode)
        options_menu.addAction(self.fit_act)

        # Native Size Action
        self.native_act = QAction("Native Size", self, checkable=True)
        self.native_act.setChecked(not self.fit_to_window)
        self.native_act.triggered.connect(self.set_native_size)
        options_menu.addAction(self.native_act)

        options_menu.addSeparator()

        # Auto-advance toggle (for Phase 6)
        self.auto_advance_act = QAction("Auto-advance on Save", self, checkable=True)
        self.auto_advance_act.setChecked(False)
        options_menu.addAction(self.auto_advance_act)

        # Zoom menu
        zoom_menu = menubar.addMenu("Zoom")

        # Standard zoom levels
        self.zoom_actions = {}
        zoom_levels = [25, 50, 75, 100, 150, 200, 300, 400]

        for level in zoom_levels:
            action = QAction(f"{level}%", self)
            action.setData(level)
            action.triggered.connect(
                lambda checked, lvl=level: self._set_zoom_level(lvl)
            )
            zoom_menu.addAction(action)
            self.zoom_actions[level] = action

        zoom_menu.addSeparator()

        # Custom zoom option
        custom_zoom_act = QAction("Custom...", self)
        custom_zoom_act.setShortcut(QKeySequence("Ctrl+G"))
        custom_zoom_act.triggered.connect(self._show_custom_zoom_dialog)
        zoom_menu.addAction(custom_zoom_act)

        zoom_menu.addSeparator()

        # Zoom In/Out actions
        zoom_in_act = QAction("Zoom In", self)
        zoom_in_act.setShortcut(QKeySequence("Ctrl+="))  # = key (no shift needed)
        zoom_in_act.triggered.connect(lambda: self._adjust_zoom(1.1))
        zoom_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("Zoom Out", self)
        zoom_out_act.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_act.triggered.connect(lambda: self._adjust_zoom(0.9))
        zoom_menu.addAction(zoom_out_act)

    def _set_zoom_level(self, percent):
        """Set zoom to a specific percentage."""
        # Switch to native size mode if in fit mode
        if self.fit_to_window:
            self.fit_to_window = False
            self.fit_act.setChecked(False)
            self.native_act.setChecked(True)

        self.zoom_factor = percent / 100.0
        self.update_image()
        self.update_zoom_display()
        self.status_bar.showMessage(f"Zoom: {percent}%", 2000)

    def _adjust_zoom(self, multiplier):
        """Adjust zoom by a multiplier (for zoom in/out)."""
        if self.fit_to_window:
            self.fit_to_window = False
            self.fit_act.setChecked(False)
            self.native_act.setChecked(True)

        self.zoom_factor *= multiplier
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        self.update_image()
        self.update_zoom_display()

    def _show_custom_zoom_dialog(self):
        """Show dialog for entering a custom zoom percentage."""
        current_percent = int(self.zoom_factor * 100)

        percent, ok = QInputDialog.getInt(
            self,
            "Custom Zoom",
            "Enter zoom percentage (10-1000):",
            current_percent,
            10,  # minimum
            1000,  # maximum
            1,  # step
        )

        if ok:
            self._set_zoom_level(percent)

    def _on_resolution_changed(self, index):
        """Handle resolution dropdown change."""
        if self.selection_locked:
            # If locked, ignore dropdown changes
            return

        size = self.resolution_combo.currentData()
        self.selection_size = size

        # Reset the overlay's selection so it redraws with new size
        if self.overlay:
            self.overlay.selection_ratios = None
            self.overlay.selection_position_ratios = None

        if size:
            # Enable set button
            self.set_btn.setEnabled(True)

            # Check if selection is larger than the original image
            if not self.original_pixmap.isNull():
                img_w = self.original_pixmap.width()
                img_h = self.original_pixmap.height()

                if size.width() > img_w or size.height() > img_h:
                    # Selection is larger than image - auto-switch to screen mode
                    # and zoom to fit the selection box
                    self._handle_oversized_selection(size, img_w, img_h)
                    return

            # In image mode, switch to native size for accurate pixel selection
            if self.size_mode == self.SIZE_MODE_IMAGE and self.fit_to_window:
                self.fit_to_window = False
                self.zoom_factor = 1.0
                self.fit_act.setChecked(False)
                self.native_act.setChecked(True)
                self.update_image()

            self.status_bar.showMessage(
                f"Selection size: {size.width()} × {size.height()}", 3000
            )
        else:
            self.set_btn.setEnabled(False)
            self.status_bar.showMessage("Please select a resolution", 3000)

        self._update_overlay()
        self._update_selection_display()

    def _handle_oversized_selection(self, size, img_w, img_h):
        """Handle when selection size is larger than the image.

        Automatically switches to screen mode and zooms the image so the
        selection box fits within the displayed image.
        """
        # Calculate the zoom factor needed to make the image fill the selection box
        # We need the image to be at least as large as the selection in both dimensions
        zoom_for_width = size.width() / img_w
        zoom_for_height = size.height() / img_h
        required_zoom = max(zoom_for_width, zoom_for_height)

        # Switch to screen mode
        self.size_mode = self.SIZE_MODE_SCREEN
        self.screen_mode_radio.setChecked(True)
        self.image_mode_radio.setChecked(False)

        # Update overlay mode
        if self.overlay:
            self.overlay.set_size_mode(YoloOverlay.SIZE_MODE_SCREEN)

        # Set zoom and switch to native size mode
        self.fit_to_window = False
        self.zoom_factor = required_zoom
        self.fit_act.setChecked(False)
        self.native_act.setChecked(True)
        self.update_image()

        # Show explanation to user
        self.status_bar.showMessage(
            f"Selection {size.width()}×{size.height()} > image {img_w}×{img_h}: "
            f"Switched to Screen mode, zoomed to {int(required_zoom * 100)}%",
            5000,
        )

        self._update_overlay()
        self._update_selection_display()

    def _on_size_mode_changed(self, button):
        """Handle size mode radio button change."""
        if self.selection_locked:
            # Restore the previous selection
            if self.size_mode == self.SIZE_MODE_IMAGE:
                self.image_mode_radio.setChecked(True)
            else:
                self.screen_mode_radio.setChecked(True)
            self.status_bar.showMessage("Clear selection first to change mode", 3000)
            return

        old_mode = self.size_mode
        if button == self.image_mode_radio:
            self.size_mode = self.SIZE_MODE_IMAGE
        else:
            self.size_mode = self.SIZE_MODE_SCREEN

        if old_mode != self.size_mode:
            # Reset selection when mode changes
            if self.overlay:
                self.overlay.selection_ratios = None
                self.overlay.selection_position_ratios = None
                self.overlay.set_size_mode(
                    YoloOverlay.SIZE_MODE_IMAGE
                    if self.size_mode == self.SIZE_MODE_IMAGE
                    else YoloOverlay.SIZE_MODE_SCREEN
                )
            self._update_overlay()
            self._update_selection_display()

            mode_name = "Image" if self.size_mode == self.SIZE_MODE_IMAGE else "Screen"
            self.status_bar.showMessage(f"Size mode: {mode_name} resolution", 3000)

    def _on_set_clicked(self):
        """Handle Set button click - lock the selection size."""
        if self.selection_size is None:
            return

        self.selection_locked = True
        self.set_btn.setEnabled(False)
        self.clear_btn.setEnabled(True)
        self.resolution_combo.setEnabled(False)
        self.image_mode_radio.setEnabled(False)
        self.screen_mode_radio.setEnabled(False)

        # Save as last used selection for restoring on new images
        self._last_selection_size = QSize(self.selection_size)
        self._last_size_mode = self.size_mode

        # Tell overlay that selection is locked (enables drawing)
        if self.overlay:
            self.overlay.set_selection_locked(True)

        mode_name = "image" if self.size_mode == self.SIZE_MODE_IMAGE else "screen"
        self.status_bar.showMessage(
            f"Selection locked: {self.selection_size.width()} × {self.selection_size.height()} ({mode_name} pixels)",
            5000,
        )

    def _on_clear_clicked(self):
        """Handle Clear button click - unlock and clear selection."""
        self.selection_locked = False
        self.selection_size = None
        self.resolution_combo.setCurrentIndex(0)  # Reset to "Please select..."
        self.resolution_combo.setEnabled(True)
        self.image_mode_radio.setEnabled(True)
        self.screen_mode_radio.setEnabled(True)
        self.set_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        if self.overlay:
            self.overlay.set_selection_locked(False)  # Disable drawing
            self.overlay.clear_selection()

        self._update_selection_display()
        self.status_bar.showMessage("Selection cleared", 3000)

    def _update_selection_display(self):
        """Update status bar with selection info."""
        if self.selection_size:
            mode_suffix = (
                " (img)" if self.size_mode == self.SIZE_MODE_IMAGE else " (scr)"
            )
            lock_suffix = " [SET]" if self.selection_locked else ""
            self.selection_label.setText(
                f"Selection: {self.selection_size.width()} × {self.selection_size.height()}{mode_suffix}{lock_suffix}"
            )
        else:
            self.selection_label.clear()

    def _create_overlay(self):
        """Create the overlay widget if it doesn't exist."""
        if self.overlay is None:
            self.overlay = YoloOverlay(self.image_label)
            self.overlay.selection_changed.connect(self._on_selection_changed)
            self.overlay.annotations_changed.connect(self._on_annotations_changed)
            # Set initial size mode
            self.overlay.set_size_mode(
                YoloOverlay.SIZE_MODE_IMAGE
                if self.size_mode == self.SIZE_MODE_IMAGE
                else YoloOverlay.SIZE_MODE_SCREEN
            )
            # Set selection locked state (drawing only allowed when locked)
            self.overlay.set_selection_locked(self.selection_locked)
            # Set current draw label if one is selected
            selected_label = self.label_panel.get_selected_label()
            if selected_label:
                self.overlay.set_draw_label(selected_label)

    def _calculate_image_offset(self):
        """Calculate the offset of the image within the label due to centering."""
        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return QPoint(0, 0)

        label_size = self.image_label.size()
        pixmap_size = displayed_pixmap.size()

        # Image is centered in the label
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2

        return QPoint(max(0, offset_x), max(0, offset_y))

    def _update_overlay(self):
        """Update overlay based on current selection size and image."""
        if self.original_pixmap.isNull():
            if self.overlay:
                self.overlay.hide()
            return

        # Get displayed image size
        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return

        if self.selection_size is None:
            if self.overlay:
                self.overlay.clear_selection()
            return

        # Create overlay if needed
        self._create_overlay()

        # Calculate the selection box size in display coordinates
        if self.size_mode == self.SIZE_MODE_IMAGE:
            # Image mode: selection_size is in original image pixels
            # Scale to current display size
            scale = displayed_pixmap.width() / self.original_pixmap.width()
            display_selection_size = QSize(
                int(self.selection_size.width() * scale),
                int(self.selection_size.height() * scale),
            )
        else:
            # Screen mode: selection_size is in screen pixels (fixed)
            display_selection_size = QSize(self.selection_size)

        # Calculate image offset within the label
        image_offset = self._calculate_image_offset()

        # Update overlay geometry to match image label
        self.overlay.setGeometry(self.image_label.rect())
        self.overlay.set_selection_size(
            display_selection_size, displayed_pixmap.size(), image_offset
        )
        self.overlay.raise_()

    def _on_selection_changed(self):
        """Handle selection box movement."""
        # Update any dependent UI (will be used more in later phases)
        pass

    def _on_label_selected(self, class_id, color, name):
        """Handle label selection from the label panel."""
        self.status_bar.showMessage(
            f"Selected label: {name} (class {class_id}) - Draw mode active", 3000
        )
        # Update overlay with selected label for drawing
        if self.overlay:
            self.overlay.set_draw_label(
                {"class_id": class_id, "color": color, "name": name}
            )

    def _on_label_deleted(self, class_id):
        """Handle label deletion - remove corresponding annotations."""
        if not self.overlay:
            return

        if self.overlay.delete_annotations_by_class(class_id):
            self.status_bar.showMessage(
                f"Removed annotations for deleted label (class {class_id})", 3000
            )
            self._set_unsaved_changes(True)

    def _on_labels_renumbered(self, id_mapping):
        """Handle label renumbering - update all label files and current annotations.

        Args:
            id_mapping: dict mapping old_class_id -> new_class_id
        """
        if not id_mapping:
            return

        # Update current overlay annotations
        if self.overlay:
            for annotation in self.overlay.annotations:
                old_id = annotation["class_id"]
                if old_id in id_mapping:
                    annotation["class_id"] = id_mapping[old_id]
            self.overlay.update()

        # Update all label files in the output directory
        output_path = self.label_panel.output_path
        if output_path and os.path.isdir(output_path):
            updated_count = self._update_all_label_files(output_path, id_mapping)
            if updated_count > 0:
                self.status_bar.showMessage(
                    f"Updated {updated_count} label file(s) with renumbered classes",
                    3000,
                )

            # Also update the YAML file
            self._update_yolo_yaml(output_path)

    def _update_all_label_files(self, output_path, id_mapping):
        """Update all .txt label files with new class IDs."""
        labels_dir = os.path.join(output_path, "labels")
        all_labels = self.label_panel.get_all_labels()
        return YoloFileHandler.update_label_files(labels_dir, id_mapping, all_labels)

    def _on_labels_changed(self):
        """Handle changes to the labels list (color, name, add, delete)."""
        if not self.overlay:
            return

        # Update the current draw label
        selected = self.label_panel.get_selected_label()
        if selected:
            self.overlay.set_draw_label(selected)
        else:
            self.overlay.set_draw_label(None)

        # Update colors/names of existing annotations to match label panel
        all_labels = {lbl["class_id"]: lbl for lbl in self.label_panel.get_all_labels()}
        annotations_updated = False

        for annotation in self.overlay.annotations:
            class_id = annotation["class_id"]
            if class_id in all_labels:
                label_info = all_labels[class_id]
                # Update color and name if they changed
                if (
                    annotation["color"] != label_info["color"]
                    or annotation["name"] != label_info["name"]
                ):
                    annotation["color"] = label_info["color"]
                    annotation["name"] = label_info["name"]
                    annotations_updated = True

        if annotations_updated:
            self.overlay.update()
            self._set_unsaved_changes(True)

    def _on_annotations_changed(self):
        """Handle changes to annotations (added, moved, resized, deleted)."""
        self._set_unsaved_changes(True)

    def _set_unsaved_changes(self, has_changes):
        """Set the unsaved changes state and update window title."""
        self.has_unsaved_changes = has_changes
        self._update_window_title()

    def _update_window_title(self):
        """Update window title with current file and unsaved indicator."""
        if not self.image_list:
            title = "YOLO Labeling Tool: Empty"
        else:
            path = self.image_list[self.current_index]
            title = f"YOLO Labeling [{self.current_index + 1}/{len(self.image_list)}]: {os.path.basename(path)}"
            if self.has_unsaved_changes:
                title = "* " + title
        self.setWindowTitle(title)

    def _check_unsaved_changes(self):
        """Check for unsaved changes and prompt user.

        Returns True if it's OK to proceed (no changes, or user chose to discard).
        Returns False if user chose to cancel.
        """
        if not self.has_unsaved_changes:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        if reply == QMessageBox.StandardButton.Save:
            self._on_save_requested()
            # Check if save was successful (changes cleared)
            return not self.has_unsaved_changes
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        else:  # Cancel
            return False

    def _on_output_path_changed(self, path):
        """Handle output path change."""
        if path:
            self.status_bar.showMessage(f"Output path: {path}", 3000)
            # Try to load classes from existing YAML file
            self._load_classes_from_yaml(path)
            # Update extractions list for the new path
            self._update_extractions_list()
            # Auto-load annotations from the first extraction if available
            self._load_existing_annotations()

    def _on_load_directory_requested(self):
        """Handle Load Directory button - reload all annotations from output directory."""
        output_path = self.label_panel.output_path
        if not output_path or not os.path.isdir(output_path):
            self.status_bar.showMessage("No valid output directory set", 3000)
            return

        # Reload classes from YAML
        self._load_classes_from_yaml(output_path)

        # Update extractions list
        self._update_extractions_list()

        # Load first extraction if available
        self._load_existing_annotations()

        self.status_bar.showMessage("Reloaded directory contents", 3000)

    def _on_refresh_files_requested(self):
        """Handle Refresh Files button - update all label files to match current class IDs."""
        output_path = self.label_panel.output_path
        if not output_path or not os.path.isdir(output_path):
            self.status_bar.showMessage("No valid output directory set", 3000)
            return

        labels_dir = os.path.join(output_path, "labels")
        if not os.path.isdir(labels_dir):
            self.status_bar.showMessage("No labels directory found", 3000)
            return

        all_labels = self.label_panel.get_all_labels()
        if not all_labels:
            self.status_bar.showMessage("No labels defined", 3000)
            return

        # Count files to update
        updated_count = 0
        error_count = 0

        try:
            for filename in os.listdir(labels_dir):
                if not filename.endswith(".txt"):
                    continue

                label_path = os.path.join(labels_dir, filename)
                try:
                    # Read, update class IDs, and rewrite the file
                    if self._refresh_single_label_file(label_path, all_labels):
                        updated_count += 1
                except Exception:
                    error_count += 1

            # Also update the YAML file
            self._update_yolo_yaml(output_path)

            if error_count > 0:
                self.status_bar.showMessage(
                    f"Refreshed {updated_count} file(s), {error_count} error(s)", 5000
                )
            else:
                self.status_bar.showMessage(
                    f"Refreshed {updated_count} label file(s)", 3000
                )

            # Reload current annotations to reflect any changes
            self._load_existing_annotations()

        except OSError as e:
            self.status_bar.showMessage(f"Error reading labels directory: {e}", 5000)

    def _refresh_single_label_file(self, label_path, all_labels):
        """Refresh a single label file to match current class definitions.

        Returns True if the file was modified.
        """
        # Load current annotations
        annotations, source_info = YoloFileHandler.load_annotations(
            label_path, all_labels, DEFAULT_COLORS
        )

        if not annotations and not source_info:
            return False

        # Rewrite with current class mappings
        YoloFileHandler.save_annotations(
            label_path, annotations, all_labels, source_info
        )
        return True

    def _load_classes_from_yaml(self, output_path):
        """Load class labels and colors from any YAML file in the directory."""
        try:
            labels, yaml_path = YoloFileHandler.load_yaml(output_path)
            if labels:
                self.label_panel.set_labels(labels)
                yaml_filename = os.path.basename(yaml_path) if yaml_path else "yaml"
                self.status_bar.showMessage(
                    f"Loaded {len(labels)} class(es) from {yaml_filename}", 3000
                )
                return True
            elif yaml_path:
                # Found a YAML file but couldn't parse classes from it
                yaml_filename = os.path.basename(yaml_path)
                self.status_bar.showMessage(
                    f"Found {yaml_filename} but no classes defined", 3000
                )
        except Exception as e:
            self.status_bar.showMessage(f"Error loading YAML: {e}", 5000)

        return False

    # Special marker for "Save new extraction" option
    SAVE_NEW_EXTRACTION = "__SAVE_NEW__"

    def _update_extractions_list(self):
        """Update the extractions dropdown with existing images for current source.

        The dropdown shows:
        - If no extractions exist: "Press Ctrl+S to extract"
        - If extractions exist: list of extraction filenames + "Save new extraction" at the end
        """
        self.extractions_combo.blockSignals(True)
        self.extractions_combo.clear()

        output_path = self.label_panel.output_path
        if not output_path or not os.path.isdir(output_path):
            self.extractions_combo.addItem("(No output path set)")
            self.extractions_combo.blockSignals(False)
            return

        if not self.image_list:
            self.extractions_combo.addItem("(No image loaded)")
            self.extractions_combo.blockSignals(False)
            return

        # Get base name of current image (without extension)
        current_path = self.image_list[self.current_index]
        base_name = os.path.splitext(os.path.basename(current_path))[0]

        # Look for extracted images matching pattern: basename_N.jpg
        images_dir = os.path.join(output_path, "images")
        extractions = []

        if os.path.isdir(images_dir):
            # Find all matching images
            import re

            pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.jpg$", re.IGNORECASE)

            try:
                for filename in os.listdir(images_dir):
                    match = pattern.match(filename)
                    if match:
                        suffix_num = int(match.group(1))
                        extractions.append((suffix_num, filename))

                # Sort by suffix number
                extractions.sort(key=lambda x: x[0])
            except OSError:
                pass

        if not extractions:
            # No extractions yet - prompt user to create first one
            self.extractions_combo.addItem("Press Ctrl+S to extract")
        else:
            # List existing extractions, auto-select the first one
            for suffix_num, filename in extractions:
                self.extractions_combo.addItem(filename, filename)
            # Add option to create a new extraction at the end
            self.extractions_combo.addItem(
                "Save new extraction", self.SAVE_NEW_EXTRACTION
            )

        # Update group box title with count
        extraction_count = len(extractions)
        self.extractions_group.setTitle(f"Extracted Images ({extraction_count})")

        self.extractions_combo.blockSignals(False)

    def _show_extractions_context_menu(self, pos):
        """Show context menu for extractions dropdown."""
        # Only show if a real extraction is selected (not placeholder or "Save new")
        data = self.extractions_combo.currentData()
        if data is None or data == self.SAVE_NEW_EXTRACTION:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Delete extraction")
        delete_action.triggered.connect(self._delete_selected_extraction)
        menu.exec(self.extractions_combo.mapToGlobal(pos))

    def _delete_selected_extraction(self):
        """Delete the currently selected extraction image and label file."""
        data = self.extractions_combo.currentData()
        if data is None or data == self.SAVE_NEW_EXTRACTION:
            return

        filename = data  # e.g., "image_0.jpg"
        base_name = os.path.splitext(filename)[0]  # e.g., "image_0"

        output_path = self.label_panel.output_path
        if not output_path:
            return

        image_path = os.path.join(output_path, "images", filename)
        label_path = os.path.join(output_path, "labels", f"{base_name}.txt")

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Extraction",
            f"Delete {filename} and its annotations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete files
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)

            self.status_bar.showMessage(f"Deleted: {filename}", 3000)

            # Refresh extractions list
            self._update_extractions_list()

            # Clear annotations since the extraction is gone
            if self.overlay:
                self.overlay.clear_annotations()

        except OSError as e:
            self.status_bar.showMessage(f"Delete failed: {e}", 5000)

    def _on_extraction_selected(self, index):
        """Handle selection of an extraction from the dropdown."""
        if index < 0:
            return

        data = self.extractions_combo.currentData()

        # Check if this is a placeholder message (no data) or "Save new extraction"
        if data is None:
            # This is the "Press Ctrl+S to extract" placeholder - do nothing
            return

        if data == self.SAVE_NEW_EXTRACTION:
            # User selected "Save new extraction" - clear annotations for new extraction
            if self.overlay:
                self.overlay.clear_annotations()
            self._set_unsaved_changes(True)
            self.status_bar.showMessage(
                "Ready to save new extraction - press Ctrl+S", 3000
            )
            return

        # data is a filename - load that extraction
        filename = data
        output_path = self.label_panel.output_path
        if not output_path:
            return

        # Load the corresponding label file
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(output_path, "labels", f"{base_name}.txt")

        try:
            current_labels = self.label_panel.get_all_labels()
            annotations, source_info = YoloFileHandler.load_annotations(
                label_path, current_labels, DEFAULT_COLORS
            )

            # Restore selection box position if source info is available
            if source_info and not self.original_pixmap.isNull():
                sel_x = source_info.get("sel_x", 0)
                sel_y = source_info.get("sel_y", 0)
                sel_w = source_info.get("sel_w", 0)
                sel_h = source_info.get("sel_h", 0)

                if sel_w > 0 and sel_h > 0:
                    # Compute selection size in original image pixels
                    sel_width_px = int(sel_w * self.original_pixmap.width())
                    sel_height_px = int(sel_h * self.original_pixmap.height())

                    # Set selection size and mode
                    self.selection_size = QSize(sel_width_px, sel_height_px)
                    self.size_mode = self.SIZE_MODE_IMAGE
                    self.image_mode_radio.setChecked(True)
                    self.screen_mode_radio.setChecked(False)

                    # Save as last used selection for restoring on new images
                    self._last_selection_size = QSize(self.selection_size)
                    self._last_size_mode = self.size_mode

                    # Lock the selection
                    self.selection_locked = True
                    self.set_btn.setEnabled(False)
                    self.clear_btn.setEnabled(True)
                    self.resolution_combo.setEnabled(False)
                    self.image_mode_radio.setEnabled(False)
                    self.screen_mode_radio.setEnabled(False)

                    # Create overlay if needed
                    self._create_overlay()

                    # Set selection ratios on overlay
                    self.overlay.selection_ratios = (sel_x, sel_y, sel_w, sel_h)
                    self.overlay.selection_position_ratios = (
                        sel_x + sel_w / 2,
                        sel_y + sel_h / 2,
                    )
                    self.overlay.set_size_mode(YoloOverlay.SIZE_MODE_IMAGE)
                    self.overlay.set_selection_locked(True)

                    self._update_overlay()
                    self._update_selection_display()

            # Load annotations
            if self.overlay:
                self.overlay.clear_annotations()
                if annotations:
                    self.overlay.annotations = annotations
                    self.overlay.update()

            self._set_unsaved_changes(False)
            self.status_bar.showMessage(
                f"Loaded extraction: {filename} ({len(annotations)} annotation(s))",
                3000,
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error loading extraction: {str(e)}", 3000)

    def _get_next_extraction_suffix(self):
        """Get the next available suffix number for a new extraction.

        Returns:
            int: Next available suffix (0, 1, 2, ...)
        """
        output_path = self.label_panel.output_path
        if not output_path or not self.image_list:
            return 0

        current_path = self.image_list[self.current_index]
        base_name = os.path.splitext(os.path.basename(current_path))[0]

        images_dir = os.path.join(output_path, "images")
        if not os.path.isdir(images_dir):
            return 0

        # Find the highest existing suffix
        import re

        pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.jpg$", re.IGNORECASE)
        max_suffix = -1

        try:
            for filename in os.listdir(images_dir):
                match = pattern.match(filename)
                if match:
                    suffix_num = int(match.group(1))
                    max_suffix = max(max_suffix, suffix_num)
        except OSError:
            pass

        return max_suffix + 1

    def _on_save_requested(self):
        """Handle save button click - save cropped image and YOLO annotations."""
        # Validate state
        if not self.image_list:
            self.status_bar.showMessage("No image to save", 3000)
            return

        output_path = self.label_panel.output_path
        if not output_path or not os.path.isdir(output_path):
            self.status_bar.showMessage("Please select a valid output directory", 3000)
            return

        if not self.selection_locked:
            self.status_bar.showMessage("Please set a selection size first", 3000)
            return

        if not self.overlay:
            self.status_bar.showMessage("No overlay available", 3000)
            return

        # Get selection rect in original image coordinates
        selection_rect_orig = self._get_selection_rect_original()
        if (
            selection_rect_orig.isNull()
            or selection_rect_orig.width() <= 0
            or selection_rect_orig.height() <= 0
        ):
            self.status_bar.showMessage("Invalid selection area", 3000)
            return

        # Get current image info
        current_path = self.image_list[self.current_index]
        base_name = os.path.splitext(os.path.basename(current_path))[0]

        # Create images subdirectory if it doesn't exist
        images_dir = os.path.join(output_path, "images")
        labels_dir = os.path.join(output_path, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        try:
            # Crop and save image
            cropped_pixmap = self._crop_image_to_selection(selection_rect_orig)
            if cropped_pixmap.isNull():
                self.status_bar.showMessage("Failed to crop image", 3000)
                return

            # Scale to exact selection size if needed (in case of rounding)
            if self.selection_size:
                cropped_pixmap = cropped_pixmap.scaled(
                    self.selection_size,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

            # Determine save name based on dropdown selection
            selected_data = self.extractions_combo.currentData()

            if selected_data is None or selected_data == self.SAVE_NEW_EXTRACTION:
                # Create new extraction with next available suffix
                suffix = self._get_next_extraction_suffix()
                save_name = f"{base_name}_{suffix}"
                is_new = True
            else:
                # Overwrite existing extraction
                save_name = os.path.splitext(selected_data)[0]  # Remove .jpg extension
                is_new = False

            # Save image
            image_path = os.path.join(images_dir, f"{save_name}.jpg")
            if not cropped_pixmap.save(image_path, "JPEG", 95):
                self.status_bar.showMessage(
                    f"Failed to save image to {image_path}", 3000
                )
                return

            # Save YOLO annotations
            annotations = self.overlay.get_annotations()
            label_path = os.path.join(labels_dir, f"{save_name}.txt")
            self._save_yolo_annotations(label_path, annotations)

            # Update yolo.yaml with class names
            self._update_yolo_yaml(output_path)

            # Clear unsaved changes
            self._set_unsaved_changes(False)

            # Update the extractions list
            _old_index = self.extractions_combo.currentIndex()
            self._update_extractions_list()

            # Select the saved extraction in the dropdown
            if is_new:
                # Find and select the newly created extraction
                for i in range(self.extractions_combo.count()):
                    if self.extractions_combo.itemData(i) == f"{save_name}.jpg":
                        self.extractions_combo.setCurrentIndex(i)
                        break
            else:
                # Re-select the same index (or find by name if index changed)
                for i in range(self.extractions_combo.count()):
                    if self.extractions_combo.itemData(i) == f"{save_name}.jpg":
                        self.extractions_combo.setCurrentIndex(i)
                        break

            # Show success message
            num_annotations = len(annotations)
            action = "Created" if is_new else "Updated"
            self.status_bar.showMessage(
                f"{action}: {save_name}.jpg with {num_annotations} annotation(s)", 3000
            )

            # Auto-advance if enabled
            if self.auto_advance_act.isChecked() and self.image_list:
                self.current_index = (self.current_index + 1) % len(self.image_list)
                self.load_image()
                self.status_bar.showMessage("Saved and advanced to next image", 3000)

        except Exception as e:
            self.status_bar.showMessage(f"Save error: {str(e)}", 5000)

    def _get_selection_rect_original(self):
        """Get selection rect in original image coordinates."""
        if self.original_pixmap.isNull() or not self.overlay:
            return QRect()

        if self.size_mode == self.SIZE_MODE_IMAGE:
            # Image mode: selection ratios directly map to original image
            ratios = self.overlay.selection_ratios
            if ratios is None:
                return QRect()

            x = int(ratios[0] * self.original_pixmap.width())
            y = int(ratios[1] * self.original_pixmap.height())
            w = int(ratios[2] * self.original_pixmap.width())
            h = int(ratios[3] * self.original_pixmap.height())

            return QRect(x, y, w, h)
        else:
            # Screen mode: need to convert from display to original
            position_ratios = self.overlay.selection_position_ratios
            if position_ratios is None or self.selection_size is None:
                return QRect()

            # Center in original image coordinates
            center_x = int(position_ratios[0] * self.original_pixmap.width())
            center_y = int(position_ratios[1] * self.original_pixmap.height())

            # Size needs to be converted from screen pixels to original pixels
            # The screen size is fixed, so we need to account for current zoom
            display_pixmap = self.image_label.pixmap()
            if display_pixmap and not display_pixmap.isNull():
                scale = self.original_pixmap.width() / display_pixmap.width()
                w = int(self.selection_size.width() * scale)
                h = int(self.selection_size.height() * scale)
            else:
                w = self.selection_size.width()
                h = self.selection_size.height()

            x = center_x - w // 2
            y = center_y - h // 2

            return QRect(x, y, w, h)

    def _crop_image_to_selection(self, rect):
        """Crop the original image to the given rect."""
        if self.original_pixmap.isNull():
            return QPixmap()

        # Ensure rect is within image bounds
        img_rect = self.original_pixmap.rect()
        rect = rect.intersected(img_rect)

        if rect.isEmpty():
            return QPixmap()

        return self.original_pixmap.copy(rect)

    def _save_yolo_annotations(self, path, annotations):
        """Save annotations in YOLO format with class mapping header and source info."""
        all_labels = self.label_panel.get_all_labels()

        # Build source info with original image metadata
        source_info = None
        if self.image_list and not self.original_pixmap.isNull():
            current_path = self.image_list[self.current_index]

            # Get selection box position as ratios of original image
            sel_x, sel_y, sel_w, sel_h = 0.0, 0.0, 0.0, 0.0
            if self.overlay:
                if self.size_mode == self.SIZE_MODE_IMAGE:
                    # Image mode: selection_ratios directly map to original image
                    if self.overlay.selection_ratios:
                        sel_x, sel_y, sel_w, sel_h = self.overlay.selection_ratios
                else:
                    # Screen mode: convert from position ratios and screen size
                    if self.overlay.selection_position_ratios and self.selection_size:
                        cx, cy = self.overlay.selection_position_ratios
                        display_pixmap = self.image_label.pixmap()
                        if display_pixmap and not display_pixmap.isNull():
                            scale = (
                                self.original_pixmap.width() / display_pixmap.width()
                            )
                            w = self.selection_size.width() * scale
                            h = self.selection_size.height() * scale
                            sel_w = w / self.original_pixmap.width()
                            sel_h = h / self.original_pixmap.height()
                            sel_x = cx - sel_w / 2
                            sel_y = cy - sel_h / 2

            source_info = {
                "filename": os.path.basename(current_path),
                "width": self.original_pixmap.width(),
                "height": self.original_pixmap.height(),
                "sel_x": sel_x,
                "sel_y": sel_y,
                "sel_w": sel_w,
                "sel_h": sel_h,
            }

        YoloFileHandler.save_annotations(path, annotations, all_labels, source_info)

    def _update_yolo_yaml(self, output_path):
        """Create or update yolo.yaml with class names and colors."""
        labels = self.label_panel.get_all_labels()
        YoloFileHandler.save_yaml(output_path, labels)

    def _load_existing_annotations(self):
        """Load the first existing extraction if available.

        This is called after image load to auto-select and load the first
        extraction from the dropdown. The dropdown has already been populated
        by _update_extractions_list().
        """
        # Clear existing annotations if overlay exists
        if self.overlay:
            self.overlay.clear_annotations()

        # Check if there are any existing extractions in the dropdown
        # The first item with actual data (not placeholder) is the first extraction
        has_extractions = False
        if self.extractions_combo.count() > 0:
            first_data = self.extractions_combo.itemData(0)
            if first_data is not None and first_data != self.SAVE_NEW_EXTRACTION:
                has_extractions = True
                # There's an existing extraction - select it (this will trigger loading)
                self.extractions_combo.setCurrentIndex(0)
                # Manually trigger the load since we blocked signals during list update
                # _on_extraction_selected will create the overlay if needed
                self._on_extraction_selected(0)

        # If no extractions but we have a last used selection, restore it unlocked
        if not has_extractions and self._last_selection_size is not None:
            self._restore_unlocked_selection()

    def _restore_unlocked_selection(self):
        """Restore the last used selection size in unlocked state.

        This allows the user to position and adjust before clicking Set.
        """
        # Restore selection size and mode from last used
        self.selection_size = QSize(self._last_selection_size)
        self.size_mode = self._last_size_mode

        # Set UI to unlocked state
        self.selection_locked = False
        self.set_btn.setEnabled(True)
        self.clear_btn.setEnabled(False)
        self.resolution_combo.setEnabled(True)
        self.image_mode_radio.setEnabled(True)
        self.screen_mode_radio.setEnabled(True)

        # Update radio buttons to match mode
        if self.size_mode == self.SIZE_MODE_IMAGE:
            self.image_mode_radio.setChecked(True)
            self.screen_mode_radio.setChecked(False)
        else:
            self.image_mode_radio.setChecked(False)
            self.screen_mode_radio.setChecked(True)

        # Handle oversized selection (selection larger than image)
        if not self.original_pixmap.isNull():
            img_w = self.original_pixmap.width()
            img_h = self.original_pixmap.height()

            if (
                self.selection_size.width() > img_w
                or self.selection_size.height() > img_h
            ):
                # Selection is larger than image - switch to screen mode and zoom
                self._handle_oversized_selection(self.selection_size, img_w, img_h)
                self._update_selection_display()
                self.status_bar.showMessage(
                    f"Selection restored: {self.selection_size.width()} × {self.selection_size.height()} (not set - position and click Set)",
                    4000,
                )
                return

        # In image mode, ensure we're at native size for accurate pixel selection
        # This matches the behavior in _on_resolution_changed
        if self.size_mode == self.SIZE_MODE_IMAGE:
            if self.fit_to_window:
                self.fit_to_window = False
                self.fit_act.setChecked(False)
                self.native_act.setChecked(True)
            # Always reset to 100% zoom for consistent selection sizing
            self.zoom_factor = 1.0

        # Create overlay if needed
        self._create_overlay()
        if self.overlay:
            self.overlay.set_size_mode(
                YoloOverlay.SIZE_MODE_IMAGE
                if self.size_mode == self.SIZE_MODE_IMAGE
                else YoloOverlay.SIZE_MODE_SCREEN
            )
            self.overlay.set_selection_locked(False)
            # Clear any previous position - will center on next update
            self.overlay.selection_ratios = None
            self.overlay.selection_position_ratios = None

        # Force image update to ensure displayed_pixmap is correct before overlay calc
        self.update_image()

        # Process pending events to let layouts settle, then update overlay again
        # This fixes timing issues on first navigation where layout isn't finalized
        QApplication.processEvents()
        self._update_overlay()
        self._update_selection_display()

        self.status_bar.showMessage(
            f"Selection restored: {self.selection_size.width()} × {self.selection_size.height()} (not set - position and click Set)",
            4000,
        )

    def toggle_fit_mode(self, checked):
        """Toggle fit-to-window mode."""
        self.fit_to_window = checked
        self.native_act.setChecked(not checked)
        self.update_image()
        self.update_zoom_display()

    def set_native_size(self):
        """Set native size mode."""
        self.fit_to_window = False
        self.fit_act.setChecked(False)
        self.native_act.setChecked(True)
        self.zoom_factor = 1.0
        self.update_image()
        self.update_zoom_display()

    def update_zoom_display(self):
        """Update status bar with current zoom percentage."""
        if self.fit_to_window:
            self.zoom_label.clear()
        else:
            zoom_percent = int(self.zoom_factor * 100)
            self.zoom_label.setText(f"Zoom: {zoom_percent}%")

    def _load_image_without_annotations(self):
        """Load and display the current image without loading annotations.

        Used during initial window setup to avoid timing issues with overlay sizing.
        """
        if not self.image_list:
            self.image_label.setText("No images to display")
            self.image_label.setPixmap(QPixmap())
            self._update_window_title()
            self.update_zoom_display()
            return

        path = self.image_list[self.current_index]
        self.original_pixmap = QPixmap(path)

        # Reset zoom when loading new image in non-fit mode
        if not self.fit_to_window:
            self.zoom_factor = 1.0

        # Clear selection box ratios when loading new image (reset position)
        if self.overlay:
            self.overlay.selection_ratios = None
            self.overlay.selection_position_ratios = None

        # Reset unsaved changes for new image
        self.has_unsaved_changes = False

        self.update_image()
        self.update_zoom_display()
        self._update_window_title()

        # Update the extractions dropdown for this image
        self._update_extractions_list()

    def load_image(self):
        """Load and display the current image."""
        self._load_image_without_annotations()

        # Load existing annotations if available
        self._load_existing_annotations()

    def update_image(self):
        """Update the displayed image based on current zoom/fit settings."""
        if not self.image_list or self.original_pixmap.isNull():
            return

        if self.fit_to_window:
            # Scale to scroll area size while keeping aspect ratio
            viewport_size = self.scroll_area.viewport().size()
            if viewport_size.width() > 0 and viewport_size.height() > 0:
                scaled = self.original_pixmap.scaled(
                    viewport_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled)
                self.image_label.adjustSize()
        else:
            # Use zoom factor for manual scaling
            new_width = int(self.original_pixmap.width() * self.zoom_factor)
            if new_width > 0:
                scaled = self.original_pixmap.scaledToWidth(
                    new_width, Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                self.image_label.adjustSize()

        # Update overlay
        self._update_overlay()

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key.Key_Escape:
            # Deselect annotation first, then close if nothing selected
            if self.overlay and self.overlay.selected_annotation is not None:
                self.overlay.selected_annotation = None
                self.overlay.update()
            else:
                self.close()
        elif event.key() == Qt.Key.Key_Delete:
            # Delete selected annotation
            if self.overlay and self.overlay.delete_selected_annotation():
                self.status_bar.showMessage("Annotation deleted", 2000)
        elif event.key() == Qt.Key.Key_Left:
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = (self.current_index - 1) % len(self.image_list)
                    self.load_image()
        elif event.key() == Qt.Key.Key_Right:
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = (self.current_index + 1) % len(self.image_list)
                    self.load_image()
        elif (
            event.key() == Qt.Key.Key_S
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            # Ctrl+S to save (Phase 5)
            self._on_save_requested()
        elif event.key() >= Qt.Key.Key_0 and event.key() <= Qt.Key.Key_9:
            # Number keys 0-9 for quick label selection
            class_id = event.key() - Qt.Key.Key_0
            if self.label_panel.select_label_by_id(class_id):
                self.status_bar.showMessage(f"Selected label {class_id}", 2000)
        elif event.key() == Qt.Key.Key_Home:
            # Jump to first image
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = 0
                    self.load_image()
        elif event.key() == Qt.Key.Key_End:
            # Jump to last image
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = len(self.image_list) - 1
                    self.load_image()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # If in fit-to-window mode, switch to native size mode first
            if self.fit_to_window:
                self.fit_to_window = False
                self.zoom_factor = 1.0
                self.fit_act.setChecked(False)
                self.native_act.setChecked(True)

            # Apply zoom
            delta = event.angleDelta().y()
            self.zoom_factor *= 1.1 if delta > 0 else 0.9
            self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
            self.update_image()
            self.update_zoom_display()
            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        """Handle window resize - triggers image update for fit mode."""
        super().resizeEvent(event)
        if hasattr(self, "fit_to_window") and self.fit_to_window:
            if not hasattr(self, "_resize_timer"):
                self._resize_timer = QTimer()
                self._resize_timer.setSingleShot(True)
                self._resize_timer.timeout.connect(self.update_image)
            self._resize_timer.start(50)

    def eventFilter(self, source, event):
        """Event filter for scroll area key events."""
        if event.type() == event.Type.KeyPress and source is self.scroll_area:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(source, event)

    def showEvent(self, event):
        """Handle window show event - initialize after window is fully visible."""
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            # Use singleShot to let the event loop finish processing the show/maximize
            # This ensures the window manager has fully processed the maximize request
            QTimer.singleShot(50, self._deferred_initialize)

    def _deferred_initialize(self):
        """Initialize after window geometry is finalized.

        This is called via timer from showEvent to ensure all layout
        calculations are complete before we try to position overlays.
        """
        # Process any pending events to ensure layout is fully complete
        QApplication.processEvents()

        # Double-check window is maximized (Linux window managers can be slow)
        # Retry up to 5 times with 100ms delays
        if not self.isMaximized() and self._maximize_attempts < 5:
            self._maximize_attempts += 1
            self.showMaximized()
            # Give window manager time to process, then re-run initialization
            QTimer.singleShot(100, self._deferred_initialize)
            return

        self._load_saved_settings()
        # Load image but skip annotations - will load after layout is settled
        self._load_image_without_annotations()
        # Load annotations after a brief delay to ensure display is finalized
        QTimer.singleShot(50, self._load_existing_annotations)

    def _load_saved_settings(self):
        """Load saved YOLO settings from the main app."""
        main_win = self.parent_app or find_main_window()
        if not main_win:
            return

        # Load output path first
        saved_path = getattr(main_win, "yolo_output_path", "")
        if saved_path:
            self.label_panel.set_output_path(saved_path)

        # Try to load labels from YAML file first (if path exists)
        # This ensures we get the latest saved classes with colors
        loaded_from_yaml = False
        if saved_path and os.path.isdir(saved_path):
            loaded_from_yaml = self._load_classes_from_yaml(saved_path)

        # Fall back to app settings if no YAML found
        if not loaded_from_yaml:
            saved_labels = getattr(main_win, "yolo_labels", [])
            if saved_labels:
                self.label_panel.set_labels(saved_labels)

        # Load resolution index
        saved_res_index = getattr(main_win, "yolo_resolution_index", 0)
        if 0 <= saved_res_index < self.resolution_combo.count():
            self.resolution_combo.setCurrentIndex(saved_res_index)

        # Load size mode
        saved_mode = getattr(main_win, "yolo_size_mode", "image")
        if saved_mode == "screen":
            self.size_mode = self.SIZE_MODE_SCREEN
            self.screen_mode_radio.setChecked(True)
            if self.overlay:
                self.overlay.set_size_mode(YoloOverlay.SIZE_MODE_SCREEN)
        else:
            self.size_mode = self.SIZE_MODE_IMAGE
            self.image_mode_radio.setChecked(True)

    def closeEvent(self, event):
        """Save settings before closing."""
        # Check for unsaved changes
        if not self._check_unsaved_changes():
            event.ignore()
            return

        main_win = self.parent_app or find_main_window()
        if main_win:
            # Save label panel state (geometry not saved since always maximized)
            main_win.yolo_labels = self.label_panel.get_all_labels()
            main_win.yolo_output_path = self.label_panel.output_path
            main_win.yolo_resolution_index = self.resolution_combo.currentIndex()
            main_win.yolo_size_mode = self.size_mode

            main_win.write_settings()
        event.accept()
