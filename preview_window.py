"""
Image preview window for viewing and navigating images.
"""

import os
from PySide6.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QLabel,
    QMessageBox,
    QApplication,
    QStatusBar,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QDialogButtonBox,
    QFileDialog,
    QMenu,
    QDockWidget,
    QTextEdit,
)
from PySide6.QtCore import Qt, QTimer, QSize, QRect
from PySide6.QtGui import QAction, QPixmap, QActionGroup, QIcon, QIntValidator
from PIL import Image

from utils import find_main_window, reset_link_list, append_to_link_list
from crop_overlay import CropOverlay


class ImagePreviewWindow(QMainWindow):
    def __init__(self, image_list, current_index, parent=None):
        super().__init__(parent)

        # 1. Establish the Main Window reference
        self.parent_app = parent
        if not self.parent_app:
            self.parent_app = find_main_window()

        # 2. Assign core data
        self.image_list = image_list
        self.current_index = current_index
        self.zoom_factor = 1.0
        self.original_pixmap = QPixmap()

        # 3. Pull settings from the Parent App (the Source of Truth)
        self.fit_to_window = (
            getattr(self.parent_app, "preview_fit_default", False)
            if self.parent_app
            else False
        )

        # Crop state
        self.crop_mode = None
        self.crop_overlay = None

        # Extract crop setting
        self.autoname_extract = (
            getattr(self.parent_app, "preview_autoname_extract", False)
            if self.parent_app
            else False
        )

        # Fixed-size crop alignment: "image" (zoom-invariant real-pixel size,
        # the useful default for producing consistently-sized extracts) or
        # "screen" (constant on-screen size; real crop size varies with zoom).
        self.crop_align_mode = (
            getattr(self.parent_app, "preview_crop_align_mode", "image")
            if self.parent_app
            else "image"
        )

        # Dedicated folder for extracted crops. Empty means "next to the
        # original". Persisted on the main window so it survives across
        # images and across closing/reopening this window.
        self.extract_save_dir = (
            getattr(self.parent_app, "preview_extract_save_dir", "")
            if self.parent_app
            else ""
        )

        # 4. UI Setup
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_context_menu)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)

        # ComfyUI info dock (LoRAs + prompt text embedded in the PNG) - only
        # populated/shown when the main window's "Show ComfyUI Info" toggle
        # is on; hidden by default and whenever there's nothing to show
        self.comfyui_dock = QDockWidget("ComfyUI Info", self)
        self.comfyui_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.comfyui_info_text = QTextEdit()
        self.comfyui_info_text.setReadOnly(True)
        self.comfyui_dock.setWidget(self.comfyui_info_text)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.comfyui_dock)
        self.comfyui_dock.hide()

        # Status bar with clickable label
        self.status_bar = QStatusBar()
        self.crop_dimensions_label = QLabel()
        self.crop_dimensions_label.setStyleSheet("QLabel { padding: 2px; }")
        self.crop_dimensions_label.mousePressEvent = self.on_crop_dimensions_clicked
        self.crop_dimensions_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.status_bar.addWidget(self.crop_dimensions_label)

        # Zoom level display
        self.zoom_label = QLabel()
        self.zoom_label.setStyleSheet("QLabel { padding: 2px; margin-left: 10px; }")
        self.status_bar.addWidget(self.zoom_label)

        # Rating display (press 1-5 to rate, 0 to clear)
        self.rating_label = QLabel()
        self.rating_label.setStyleSheet(
            "QLabel { padding: 2px; margin-left: 10px; color: #e8b923; }"
        )
        self.rating_label.setToolTip("Press 1-5 to rate this image, 0 to clear")
        self.status_bar.addWidget(self.rating_label)

        # Extract save-folder display
        self.save_dir_label = QLabel()
        self.save_dir_label.setStyleSheet("QLabel { padding: 2px; margin-left: 10px; }")
        self.status_bar.addPermanentWidget(self.save_dir_label)
        self._update_save_dir_label()

        self.setStatusBar(self.status_bar)

        # 5. Setup Menu and Sync Checkbox
        self.setup_menu()
        if self.fit_to_window:
            if hasattr(self, "fit_act"):
                self.fit_act.setChecked(True)
        else:
            if hasattr(self, "native_act"):
                self.native_act.setChecked(True)

        # 6. Handle Window Geometry BEFORE loading image
        saved_geo = (
            getattr(self.parent_app, "preview_geometry", None)
            if self.parent_app
            else None
        )
        if saved_geo:
            self.restoreGeometry(saved_geo)
        else:
            # Fallback to a large window size based on available screen space
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.resize(int(screen_geo.width() * 0.8), int(screen_geo.height() * 0.9))

        # 7. Load Image AFTER window is sized
        # Use QTimer to ensure window is fully rendered before loading
        QTimer.singleShot(100, self.load_image)

        # 8. Final Event Configuration
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scroll_area.installEventFilter(self)

    def setup_menu(self):
        menubar = self.menuBar()
        options_menu = menubar.addMenu("Options")

        # Fit / Native are mutually exclusive — use an action group so Qt
        # manages the checked state automatically.
        self.view_action_group = QActionGroup(self)
        self.view_action_group.setExclusive(True)

        self.fit_act = QAction("Fit to Window", self, checkable=True)
        self.fit_act.triggered.connect(self.toggle_fit_mode)
        self.view_action_group.addAction(self.fit_act)
        options_menu.addAction(self.fit_act)

        self.native_act = QAction("Native Size", self, checkable=True)
        self.native_act.triggered.connect(self.set_native_size)
        self.view_action_group.addAction(self.native_act)
        options_menu.addAction(self.native_act)

        options_menu.addSeparator()

        # Autoname Extract Action
        self.autoname_extract_act = QAction("Autoname Extract", self, checkable=True)
        self.autoname_extract_act.setChecked(self.autoname_extract)
        self.autoname_extract_act.triggered.connect(self.toggle_autoname_extract)
        options_menu.addAction(self.autoname_extract_act)

        # Enhance Menu
        enhance_menu = menubar.addMenu("Enhance")

        sharpen_act = QAction("Sharpen...", self)
        sharpen_act.triggered.connect(lambda: self.open_enhance_dialog("sharpen"))
        enhance_menu.addAction(sharpen_act)

        denoise_act = QAction("Denoise...", self)
        denoise_act.triggered.connect(lambda: self.open_enhance_dialog("denoise"))
        enhance_menu.addAction(denoise_act)

        brightness_act = QAction("Brightness/Contrast...", self)
        brightness_act.triggered.connect(
            lambda: self.open_enhance_dialog("brightness_contrast")
        )
        enhance_menu.addAction(brightness_act)

        auto_levels_act = QAction("Auto Levels...", self)
        auto_levels_act.triggered.connect(
            lambda: self.open_enhance_dialog("auto_levels")
        )
        enhance_menu.addAction(auto_levels_act)

        saturation_act = QAction("Color Saturation...", self)
        saturation_act.triggered.connect(lambda: self.open_enhance_dialog("saturation"))
        enhance_menu.addAction(saturation_act)

        resize_act = QAction("Resize...", self)
        resize_act.triggered.connect(lambda: self.open_enhance_dialog("resize"))
        enhance_menu.addAction(resize_act)

        enhance_menu.addSeparator()

        # AI Model (Upscale / Denoise / Restore)
        ai_model_act = QAction("AI Model...", self)
        ai_model_act.triggered.connect(lambda: self.open_enhance_dialog("ai_upscale"))
        enhance_menu.addAction(ai_model_act)

        # Crop Menu
        crop_menu = menubar.addMenu("Crop")

        # Create action group for mutual exclusivity
        self.crop_action_group = QActionGroup(self)
        self.crop_action_group.setExclusive(True)

        # Rectangle crop
        rect_act = QAction("Rectangle", self, checkable=True)
        rect_act.triggered.connect(lambda: self.set_crop_mode("rectangle"))
        self.crop_action_group.addAction(rect_act)
        crop_menu.addAction(rect_act)

        # Square crop (free-resize, 1:1 constrained)
        square_act = QAction("Square", self, checkable=True)
        square_act.triggered.connect(lambda: self.set_crop_mode("square"))
        self.crop_action_group.addAction(square_act)
        crop_menu.addAction(square_act)

        crop_menu.addSeparator()

        # Fixed size crops
        for size in ["1024x1024", "768x768", "640x640", "512x512"]:
            act = QAction(size, self, checkable=True)
            act.triggered.connect(lambda checked, s=size: self.set_crop_mode(s))
            self.crop_action_group.addAction(act)
            crop_menu.addAction(act)

        crop_menu.addSeparator()

        # Fixed-size alignment: real image pixels (zoom-invariant) vs a
        # constant number of on-screen pixels.
        align_menu = crop_menu.addMenu("Fixed Size Aligns To")
        self.align_action_group = QActionGroup(self)
        self.align_action_group.setExclusive(True)

        align_image_act = QAction("Image (constant real size)", self, checkable=True)
        align_image_act.setChecked(self.crop_align_mode == "image")
        align_image_act.triggered.connect(lambda: self._set_crop_align_mode("image"))
        self.align_action_group.addAction(align_image_act)
        align_menu.addAction(align_image_act)

        align_screen_act = QAction(
            "Screen (constant on-screen size)", self, checkable=True
        )
        align_screen_act.setChecked(self.crop_align_mode == "screen")
        align_screen_act.triggered.connect(lambda: self._set_crop_align_mode("screen"))
        self.align_action_group.addAction(align_screen_act)
        align_menu.addAction(align_screen_act)

        crop_menu.addSeparator()

        # Clear crop
        clear_act = QAction("Clear Crop", self)
        clear_act.triggered.connect(self.clear_crop)
        crop_menu.addAction(clear_act)

        crop_menu.addSeparator()

        # Extract crop to new file (original untouched)
        self.extract_act = QAction("Extract Crop to New File", self)
        self.extract_act.setShortcut("Ctrl+E")
        self.extract_act.triggered.connect(self.extract_crop)
        crop_menu.addAction(self.extract_act)

        # Save crop over original
        self.save_crop_act = QAction("Save Crop over Original", self)
        self.save_crop_act.setShortcut("Ctrl+S")
        self.save_crop_act.triggered.connect(self.save_crop)
        crop_menu.addAction(self.save_crop_act)

        crop_menu.addSeparator()

        set_save_dir_act = QAction("Set Extract Save Folder...", self)
        set_save_dir_act.triggered.connect(self._choose_extract_save_dir)
        crop_menu.addAction(set_save_dir_act)

        clear_save_dir_act = QAction("Clear Extract Save Folder", self)
        clear_save_dir_act.triggered.connect(self._clear_extract_save_dir)
        crop_menu.addAction(clear_save_dir_act)

    def toggle_fit_mode(self, checked):
        self.fit_to_window = checked
        self.update_image()
        self.update_zoom_display()

        # Update the main window's setting
        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_fit_default = checked
            main_win.preview_geometry = self.saveGeometry()
            main_win.write_settings()

    def set_native_size(self):
        self.fit_to_window = False
        self.zoom_factor = 1.0
        self.update_image()
        self.update_zoom_display()

        # Update the main window's setting
        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_fit_default = False
            main_win.preview_geometry = self.saveGeometry()
            main_win.write_settings()

    def toggle_autoname_extract(self, checked):
        """Toggle autoname extract mode."""
        self.autoname_extract = checked

        # Update the main window's setting
        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_autoname_extract = checked
            main_win.write_settings()

    def set_crop_mode(self, mode):
        """Set crop mode and show overlay."""
        self.crop_mode = mode
        self.update_crop_overlay()

    def _set_crop_align_mode(self, mode):
        """Switch whether fixed-size crop boxes track real image pixels or screen pixels."""
        self.crop_align_mode = mode
        if self.crop_mode:
            self.update_crop_overlay()

        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_crop_align_mode = mode
            main_win.write_settings()

    def _choose_extract_save_dir(self):
        """Prompt for a dedicated folder to save extracted crops into."""
        start = self.extract_save_dir
        if not start and self.image_list:
            start = os.path.dirname(self.image_list[self.current_index])
        start = start or os.path.expanduser("~")

        path = QFileDialog.getExistingDirectory(
            self, "Select Extract Save Folder", start
        )
        if not path:
            return

        self.extract_save_dir = path
        self._update_save_dir_label()

        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_extract_save_dir = path
            main_win.write_settings()

    def _clear_extract_save_dir(self):
        """Revert extracted crops to saving next to the original image."""
        self.extract_save_dir = ""
        self._update_save_dir_label()

        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_extract_save_dir = ""
            main_win.write_settings()

    def _update_save_dir_label(self):
        if self.extract_save_dir:
            self.save_dir_label.setText(f"Save to: {self.extract_save_dir}")
        else:
            self.save_dir_label.setText("Save to: (next to original)")

    def clear_crop(self):
        """Clear crop selection."""
        self.crop_mode = None

        # Uncheck all crop actions
        if hasattr(self, "crop_action_group"):
            for action in self.crop_action_group.actions():
                action.setChecked(False)

        if self.crop_overlay:
            self.crop_overlay.set_crop_mode(None, QSize())

    def update_crop_overlay(self):
        """Update or create crop overlay based on current mode."""
        if not self.crop_mode or self.original_pixmap.isNull():
            if self.crop_overlay:
                self.crop_overlay.hide()
            self.update_crop_dimensions_display()
            return

        # Get the actual displayed image size
        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return

        image_size = displayed_pixmap.size()

        # Create overlay if it doesn't exist
        if not self.crop_overlay:
            self.crop_overlay = CropOverlay(self.image_label)
            # Connect overlay movement to dimension display update
            self.crop_overlay.installEventFilter(self)

        # Resize overlay to match image label
        self.crop_overlay.setGeometry(self.image_label.rect())
        self.crop_overlay.set_crop_mode(
            self.crop_mode,
            image_size,
            native_size=self.original_pixmap.size(),
            align_mode=self.crop_align_mode,
        )
        self.crop_overlay.raise_()
        self.update_crop_dimensions_display()

    def update_crop_dimensions_display(self):
        """Update status bar with crop box dimensions in original image coordinates."""
        if not self.crop_mode or not self.crop_overlay or self.original_pixmap.isNull():
            self.crop_dimensions_label.clear()
            return

        # Get crop rectangle in displayed coordinates
        crop_rect = self.crop_overlay.get_crop_rect()
        if crop_rect.isNull() or crop_rect.isEmpty():
            self.crop_dimensions_label.clear()
            return

        # Get the displayed pixmap to calculate scale factor
        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return

        # Calculate scale factor between displayed and original image
        scale_x = self.original_pixmap.width() / displayed_pixmap.width()
        scale_y = self.original_pixmap.height() / displayed_pixmap.height()

        # Convert crop dimensions to original image coordinates (no offset needed for dimensions)
        orig_width = int(crop_rect.width() * scale_x)
        orig_height = int(crop_rect.height() * scale_y)

        # Display in status bar label (clickable)
        self.crop_dimensions_label.setText(
            f"Crop: {orig_width} × {orig_height} (click to edit)"
        )

    def update_zoom_display(self):
        """Update status bar with current zoom percentage."""
        if self.fit_to_window:
            self.zoom_label.clear()
        else:
            zoom_percent = int(self.zoom_factor * 100)
            self.zoom_label.setText(f"Zoom: {zoom_percent}%")

    def on_crop_dimensions_clicked(self, event):
        """Handle click on crop dimensions label to manually edit dimensions."""
        if not self.crop_mode or not self.crop_overlay or self.original_pixmap.isNull():
            return

        # Get current crop dimensions in original image coordinates
        crop_rect = self.crop_overlay.get_crop_rect()
        if crop_rect.isNull() or crop_rect.isEmpty():
            return

        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return

        scale_x = self.original_pixmap.width() / displayed_pixmap.width()
        scale_y = self.original_pixmap.height() / displayed_pixmap.height()

        current_width = int(crop_rect.width() * scale_x)
        current_height = int(crop_rect.height() * scale_y)

        # Show dialog to edit dimensions
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Crop Dimensions")
        layout = QVBoxLayout()

        # Width input
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        width_input = QLineEdit(str(current_width))
        width_input.setValidator(QIntValidator(1, self.original_pixmap.width()))
        width_layout.addWidget(width_input)
        layout.addLayout(width_layout)

        # Height input
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        height_input = QLineEdit(str(current_height))
        height_input.setValidator(QIntValidator(1, self.original_pixmap.height()))
        height_layout.addWidget(height_input)
        layout.addLayout(height_layout)

        # Info label
        info_label = QLabel(
            f"Max dimensions: {self.original_pixmap.width()} × {self.original_pixmap.height()}"
        )
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                new_width = int(width_input.text())
                new_height = int(height_input.text())

                # Validate dimensions
                if new_width < 1 or new_height < 1:
                    QMessageBox.warning(
                        self,
                        "Invalid Dimensions",
                        "Width and height must be at least 1 pixel.",
                    )
                    return

                if (
                    new_width > self.original_pixmap.width()
                    or new_height > self.original_pixmap.height()
                ):
                    QMessageBox.warning(
                        self,
                        "Invalid Dimensions",
                        f"Dimensions cannot exceed image size ({self.original_pixmap.width()} × {self.original_pixmap.height()}).",
                    )
                    return

                # Convert to displayed coordinates
                new_display_width = int(new_width / scale_x)
                new_display_height = int(new_height / scale_y)

                # Keep top-left corner fixed, adjust bottom-right
                new_rect = QRect(
                    crop_rect.x(), crop_rect.y(), new_display_width, new_display_height
                )

                # Ensure crop doesn't go outside image bounds
                if new_rect.right() > displayed_pixmap.width():
                    new_rect.setRight(displayed_pixmap.width())
                if new_rect.bottom() > displayed_pixmap.height():
                    new_rect.setBottom(displayed_pixmap.height())

                # Update the crop overlay
                self.crop_overlay.crop_rect = new_rect
                self.crop_overlay._update_ratios_from_rect()
                self.crop_overlay.update()
                self.update_crop_dimensions_display()

            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Input", "Please enter valid integer dimensions."
                )
                return

    def load_image(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            self.original_pixmap = QPixmap()
            self.image_label.setStyleSheet(
                "QLabel { color: red; font-size: 24px; font-weight: bold; }"
            )
            # setPixmap()/setText() are mutually exclusive on QLabel - whichever
            # is called last wins, so clear the pixmap before setting the text
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("NO IMAGES TO DISPLAY")
            self.setWindowTitle("Preview: Empty")
            self.update_zoom_display()
            self.comfyui_dock.hide()
            self.rating_label.setText("")
            return

        self.image_label.setStyleSheet("")
        path = self.image_list[self.current_index]
        self.setWindowTitle(
            f"Preview [{self.current_index + 1}/{len(self.image_list)}]: {os.path.basename(path)}"
        )
        self.original_pixmap = QPixmap(path)
        self._update_comfyui_panel(path)
        self._update_rating_label(path)

        # Reset zoom when loading new image in non-fit mode
        if not self.fit_to_window:
            self.zoom_factor = 1.0

        # Clear crop when loading new image
        self.clear_crop()

        self.update_image()
        self.update_zoom_display()

    def _update_comfyui_panel(self, path):
        """Populate/show the ComfyUI info dock, or hide it if disabled or empty."""
        main_win = self.parent_app
        enabled = bool(
            main_win
            and getattr(main_win, "comfyui_info_act", None)
            and main_win.comfyui_info_act.isChecked()
        )
        if not enabled:
            self.comfyui_dock.hide()
            return

        info = None
        if not os.environ.get("IMG_SORT_NO_COMFYUI"):
            from comfyui_metadata import extract_comfyui_info

            info = extract_comfyui_info(path)
        if not info:
            from image_provenance import extract_source_info

            source_info = extract_source_info(path)
            if source_info:
                lines = ["Downloaded image (no ComfyUI workflow data)", ""]
                field_labels = (
                    ("Source", "Source"),
                    ("SourcePage", "Source Page"),
                    ("Query", "Query"),
                    ("Subject", "Subject"),
                )
                for key, label in field_labels:
                    if source_info.get(key):
                        lines.append(f"{label}: {source_info[key]}")
                self.comfyui_info_text.setPlainText("\n".join(lines))
            else:
                self.comfyui_info_text.setPlainText(
                    "No ComfyUI metadata found in this image."
                )
            self.comfyui_dock.show()
            return

        lines = []
        lines.append(f"Model: {info.get('model_name') or '(not detected)'}")
        lines.append("")
        if info["loras"]:
            lines.append("LoRAs:")
            for lora in info["loras"]:
                strengths = []
                if lora["strength_model"] is not None:
                    strengths.append(f"model {lora['strength_model']}")
                if lora["strength_clip"] is not None:
                    strengths.append(f"clip {lora['strength_clip']}")
                suffix = f" ({', '.join(strengths)})" if strengths else ""
                lines.append(f"  - {lora['name']}{suffix}")
        else:
            lines.append("LoRAs: none found")

        lines.append("")
        lines.append("Positive Prompt:")
        lines.append(info["positive_prompt"] or "(none found)")

        if info["negative_prompt"]:
            lines.append("")
            lines.append("Negative Prompt:")
            lines.append(info["negative_prompt"])

        if info["other_prompts"]:
            lines.append("")
            lines.append("Other Text:")
            for text in info["other_prompts"]:
                lines.append(f"  - {text}")

        self.comfyui_info_text.setPlainText("\n".join(lines))
        self.comfyui_dock.show()

    def update_image(self):
        if not self.image_list or self.original_pixmap.isNull():
            return

        if self.fit_to_window:
            # Scale to scroll area size while keeping AspectRatio
            # Use viewport size for more accurate fitting
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

        # Update crop overlay if active
        if self.crop_mode:
            self.update_crop_overlay()

    def delete_current_image(self, permanent=False):
        """Delete the current image.

        Args:
            permanent: If True, delete permanently (Shift+Delete). If False, allow undo.
        """
        if not self.image_list:
            return

        path_to_delete = self.image_list[self.current_index]
        main_win = self.parent_app or find_main_window()

        # Read the confirmation setting safely
        should_confirm = True  # Default for safety
        if main_win and hasattr(main_win, "conf_delete_act"):
            should_confirm = main_win.conf_delete_act.isChecked()

        do_delete = True
        if should_confirm:
            delete_type = "permanently delete" if permanent else "delete"
            reply = QMessageBox.question(
                self,
                "Delete Image",
                f"Are you sure you want to {delete_type} {os.path.basename(path_to_delete)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            do_delete = reply == QMessageBox.StandardButton.Yes

        if do_delete:
            try:
                # Start a new deletion batch for undo (clears previous undo state)
                if (
                    not permanent
                    and main_win
                    and hasattr(main_win, "delete_undo_manager")
                ):
                    main_win.delete_undo_manager.start_new_deletion()

                # 1. Physical Delete or store for undo
                if os.path.exists(path_to_delete):
                    if permanent:
                        os.remove(path_to_delete)
                    else:
                        # Store for undo (moves to temp location)
                        if main_win and hasattr(main_win, "delete_undo_manager"):
                            main_win.delete_undo_manager.store_for_undo(path_to_delete)
                        else:
                            os.remove(path_to_delete)

                # 2. UI SYNC: Update the main window's thumbnail list
                if main_win and hasattr(main_win, "remove_thumbnails_by_path"):
                    main_win.remove_thumbnails_by_path([path_to_delete])

                # 3. Update Preview state
                self.image_list.pop(self.current_index)
                if self.image_list:
                    # Keep index in bounds
                    if self.current_index >= len(self.image_list):
                        self.current_index = len(self.image_list) - 1
                    self.load_image()
                else:
                    # No more images
                    self.load_image()

                # Show status message
                if main_win and hasattr(main_win, "status_bar"):
                    if permanent:
                        main_win.status_bar.showMessage(
                            "Permanently deleted image", 3000
                        )
                    else:
                        main_win.status_bar.showMessage(
                            "Deleted image - Ctrl+Z to undo", 3000
                        )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete: {e}")

    def rotate_current_image(self):
        """Rotate current image 90° clockwise and update crop box if present."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        current_path = self.image_list[self.current_index]

        try:
            # Load and rotate image
            img = Image.open(current_path)

            # Store original dimensions for crop transformation
            orig_width, orig_height = img.size

            # Rotate 90° clockwise
            rotated = img.rotate(-90, expand=True)

            # Save back to file
            img_format = img.format
            if img_format in ["JPEG", "JPG"]:
                rotated.save(current_path, format=img_format, quality=95, optimize=True)
            else:
                rotated.save(current_path, format=img_format)

            # Transform crop box if in crop mode
            if self.crop_mode and self.crop_overlay and self.crop_overlay.crop_ratios:
                # Get current crop ratios (relative to old image dimensions)
                x_ratio, y_ratio, w_ratio, h_ratio = self.crop_overlay.crop_ratios

                # Transform ratios for 90° clockwise rotation:
                # old (x, y) in portrait becomes (1-y-h, x) in landscape
                # old width becomes new height, old height becomes new width
                new_x_ratio = 1.0 - y_ratio - h_ratio
                new_y_ratio = x_ratio
                new_w_ratio = h_ratio  # old height becomes new width
                new_h_ratio = w_ratio  # old width becomes new height

                # Store transformed ratios
                self.crop_overlay.crop_ratios = (
                    new_x_ratio,
                    new_y_ratio,
                    new_w_ratio,
                    new_h_ratio,
                )

            # Reload the image in preview
            self.load_image()

            # Update main window thumbnail if accessible
            main_win = self.parent_app or find_main_window()
            if main_win and hasattr(main_win, "thumb_list"):
                # Find and update the thumbnail for this image
                thumb_list = main_win.thumb_list
                for i in range(thumb_list.count()):
                    item = thumb_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == current_path:
                        # Clear and reload icon
                        item.setIcon(QIcon())
                        thumb_list.scheduleDelayedItemsLayout()
                        QApplication.processEvents()
                        item.setIcon(QIcon(current_path))
                        thumb_list.viewport().update()
                        break

            # Show status
            if main_win and hasattr(main_win, "status_bar"):
                main_win.status_bar.showMessage("Rotated image 90° clockwise", 2000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not rotate image:\n{e}")

    def generate_extract_filename(self, original_path):
        """Generate a unique filename for extracted crop with _e0, _e1, etc. suffix.

        Used when saving next to the original — the suffix distinguishes the
        extract from its sibling source file in the same folder.
        """
        directory = os.path.dirname(original_path)
        basename = os.path.basename(original_path)
        name, ext = os.path.splitext(basename)

        # Find next available number
        n = 0
        while True:
            new_name = f"{name}_e{n}{ext}"
            new_path = os.path.join(directory, new_name)
            if not os.path.exists(new_path):
                return new_path
            n += 1

    def _resolve_extract_target(self, original_path):
        """Return a unique destination path for the extracted crop.

        Dedicated save folder (self.extract_save_dir): keeps the original's
        filename, de-duplicating with a "_N" suffix on collision (matching
        the app's smart_transfer convention) since there's no longer a
        same-named sibling source file to distinguish from.

        No dedicated folder: falls back to the existing next-to-original
        "_eN" naming.
        """
        if self.extract_save_dir and os.path.isdir(self.extract_save_dir):
            basename = os.path.basename(original_path)
            name, ext = os.path.splitext(basename)
            candidate = os.path.join(self.extract_save_dir, basename)
            n = 1
            while os.path.exists(candidate):
                candidate = os.path.join(self.extract_save_dir, f"{name}_{n}{ext}")
                n += 1
            return candidate

        return self.generate_extract_filename(original_path)

    def extract_crop(self):
        """Extract the cropped area to a new file, leaving the original intact."""
        if not self.crop_mode or not self.crop_overlay:
            return

        # Get crop rectangle in displayed image coordinates
        crop_rect = self.crop_overlay.get_crop_rect()
        if crop_rect.isNull() or crop_rect.isEmpty():
            return

        # Get the displayed pixmap to calculate scale factor
        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return

        # Calculate the offset of the pixmap within the label (due to center alignment)
        label_size = self.image_label.size()
        pixmap_size = displayed_pixmap.size()
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2

        # Adjust crop rect to account for pixmap offset within label
        adjusted_x = crop_rect.x() - offset_x
        adjusted_y = crop_rect.y() - offset_y

        # Calculate scale factor between displayed and original image
        scale_x = self.original_pixmap.width() / displayed_pixmap.width()
        scale_y = self.original_pixmap.height() / displayed_pixmap.height()

        # Convert crop rectangle to original image coordinates
        orig_x = int(adjusted_x * scale_x)
        orig_y = int(adjusted_y * scale_y)
        orig_width = int(crop_rect.width() * scale_x)
        orig_height = int(crop_rect.height() * scale_y)

        # Get current image path
        if not self.image_list:
            return
        image_path = self.image_list[self.current_index]

        # Generate automatic filename
        auto_filename = self._resolve_extract_target(image_path)

        # Determine final filename
        if self.autoname_extract:
            # Use automatic name without prompting
            extract_path = auto_filename
        else:
            # Prompt user for filename
            extract_path, _ = QFileDialog.getSaveFileName(
                self,
                "Extract to File",
                auto_filename,
                "Images (*.png *.jpg *.jpeg *.gif *.webp *.avif *.tiff *.bmp *.tn3);;All Files (*)",
            )

            if not extract_path:
                # User cancelled
                return

        try:
            # Load original image with PIL
            with Image.open(image_path) as img:
                # Crop the image
                cropped = img.crop(
                    (orig_x, orig_y, orig_x + orig_width, orig_y + orig_height)
                )

                # Save to new file
                # Determine format from extension
                save_format = None
                ext = os.path.splitext(extract_path)[1].lower()
                if ext in [".jpg", ".jpeg"]:
                    save_format = "JPEG"
                    cropped.save(
                        extract_path, format=save_format, quality=95, optimize=True
                    )
                elif ext == ".png":
                    save_format = "PNG"
                    cropped.save(extract_path, format=save_format)
                else:
                    # Let PIL auto-detect format
                    cropped.save(extract_path)

            # Show success message in preview window's own status bar
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    f"Extracted crop to: {os.path.basename(extract_path)}", 5000
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not extract crop:\n{e}")

    def save_crop(self):
        """Save the cropped area to the original file."""
        if not self.crop_mode or not self.crop_overlay:
            return

        # Get crop rectangle in displayed image coordinates
        crop_rect = self.crop_overlay.get_crop_rect()
        if crop_rect.isNull() or crop_rect.isEmpty():
            return

        # Get the displayed pixmap to calculate scale factor
        displayed_pixmap = self.image_label.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return

        # Calculate the offset of the pixmap within the label (due to center alignment)
        label_size = self.image_label.size()
        pixmap_size = displayed_pixmap.size()
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2

        # Adjust crop rect to account for pixmap offset within label
        adjusted_x = crop_rect.x() - offset_x
        adjusted_y = crop_rect.y() - offset_y

        # Calculate scale factor between displayed and original image
        scale_x = self.original_pixmap.width() / displayed_pixmap.width()
        scale_y = self.original_pixmap.height() / displayed_pixmap.height()

        # Convert crop rectangle to original image coordinates
        orig_x = int(adjusted_x * scale_x)
        orig_y = int(adjusted_y * scale_y)
        orig_width = int(crop_rect.width() * scale_x)
        orig_height = int(crop_rect.height() * scale_y)

        # Get current image path
        if not self.image_list:
            return
        image_path = self.image_list[self.current_index]

        # Check confirmation setting
        main_win = self.parent_app or find_main_window()
        should_confirm = True
        if main_win and hasattr(main_win, "conf_delete_act"):
            should_confirm = main_win.conf_delete_act.isChecked()

        do_save = True
        if should_confirm:
            reply = QMessageBox.question(
                self,
                "Confirm Crop",
                f"This will overwrite the original file:\n{os.path.basename(image_path)}\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            do_save = reply == QMessageBox.StandardButton.Yes

        if do_save:
            try:
                # Load original image with PIL
                with Image.open(image_path) as img:
                    # Crop the image
                    cropped = img.crop(
                        (orig_x, orig_y, orig_x + orig_width, orig_y + orig_height)
                    )

                    # Save back to original file
                    cropped.save(image_path)

                # Reload the image
                self.original_pixmap = QPixmap(image_path)
                self.clear_crop()
                self.update_image()

                # Show success message
                if main_win:
                    main_win.status_bar.showMessage(
                        f"Cropped and saved: {os.path.basename(image_path)}", 3000
                    )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Could not save cropped image:\n{e}"
                )

    def open_enhance_dialog(self, operation=None):
        """Open the enhancement dialog for the current image."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        from enhance_dialog import EnhanceDialog

        image_path = self.image_list[self.current_index]
        dialog = EnhanceDialog([image_path], self, preset_operation=operation)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Reload the image in preview
            self.load_image()

            # Update main window thumbnail
            main_win = self.parent_app or find_main_window()
            if main_win and hasattr(main_win, "thumb_list"):
                thumb_list = main_win.thumb_list
                for i in range(thumb_list.count()):
                    item = thumb_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == image_path:
                        item.setIcon(QIcon())
                        thumb_list.scheduleDelayedItemsLayout()
                        QApplication.processEvents()
                        item.setIcon(QIcon(image_path))
                        thumb_list.viewport().update()
                        break

            if main_win and hasattr(main_win, "status_bar"):
                main_win.status_bar.showMessage(
                    f"Enhanced: {os.path.basename(image_path)}", 3000
                )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape or (
            event.key() == Qt.Key.Key_W
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.close()
        elif (
            event.key() == Qt.Key.Key_R
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            # Ctrl+R to rotate current image
            self.rotate_current_image()
        elif event.key() == Qt.Key.Key_Left:
            if self.image_list:
                self.current_index = (self.current_index - 1) % len(self.image_list)
                self.load_image()
        elif event.key() == Qt.Key.Key_Right:
            if self.image_list:
                self.current_index = (self.current_index + 1) % len(self.image_list)
                self.load_image()
        elif event.key() == Qt.Key.Key_Delete:
            # Shift+Delete = permanent delete (no undo)
            permanent = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.delete_current_image(permanent=permanent)
        elif event.key() == Qt.Key.Key_Home:
            # Jump to first image
            if self.image_list:
                self.current_index = 0
                self.load_image()
        elif event.key() == Qt.Key.Key_End:
            # Jump to last image
            if self.image_list:
                self.current_index = len(self.image_list) - 1
                self.load_image()
        elif Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_5:
            self.rate_current_image(event.key() - Qt.Key.Key_0)
        elif event.key() == Qt.Key.Key_0:
            self.rate_current_image(None)
        else:
            super().keyPressEvent(event)

    def rate_current_image(self, rating):
        """Set (or clear, if rating is None) the rating for the current image.

        Rating auto-advances to the next image, for quickly working through
        a batch; clearing stays put.
        """
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        path = self.image_list[self.current_index]

        from rating_manager import get_rating_manager

        get_rating_manager().set_rating(path, rating)
        self._update_rating_label(path)

        if self.parent_app and hasattr(self.parent_app, "thumb_list"):
            self.parent_app.thumb_list.refresh_rating_display_for_path(path)

        if rating:
            self.status_bar.showMessage(
                f"Rated {os.path.basename(path)} {rating}/5", 2000
            )
            if len(self.image_list) > 1:
                self.current_index = (self.current_index + 1) % len(self.image_list)
                self.load_image()
        else:
            self.status_bar.showMessage(
                f"Cleared rating for {os.path.basename(path)}", 2000
            )

    def _update_rating_label(self, path):
        show = bool(
            self.parent_app
            and getattr(self.parent_app, "preview_rating_display_act", None)
            and self.parent_app.preview_rating_display_act.isChecked()
        )
        if not show or not path:
            self.rating_label.setText("")
            self.rating_label.setVisible(False)
            return

        from rating_manager import format_stars, get_rating_manager

        rating = get_rating_manager().get_rating(path)
        self.rating_label.setVisible(True)
        self.rating_label.setText(format_stars(rating) if rating else "")

    def resizeEvent(self, event):
        """Handle preview window resize - triggers image update for fit mode."""
        super().resizeEvent(event)
        if hasattr(self, "fit_to_window") and self.fit_to_window:
            # Use a timer to debounce resize events
            if not hasattr(self, "_resize_timer"):
                self._resize_timer = QTimer()
                self._resize_timer.setSingleShot(True)
                self._resize_timer.timeout.connect(self.update_image)
            self._resize_timer.start(50)

    def wheelEvent(self, event):
        # Check for Ctrl modifier for zoom functionality
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # If in fit-to-window mode, switch to native size mode first
            if self.fit_to_window:
                self.fit_to_window = False
                self.zoom_factor = 1.0
                if hasattr(self, "fit_act"):
                    self.fit_act.setChecked(False)
                if hasattr(self, "native_act"):
                    self.native_act.setChecked(True)

            # Apply zoom
            delta = event.angleDelta().y()
            self.zoom_factor *= 1.1 if delta > 0 else 0.9
            self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
            self.update_image()
            self.update_zoom_display()
            event.accept()
        else:
            # Pass to scroll area for normal scrolling
            super().wheelEvent(event)

    def show_context_menu(self, position):
        """Show right-click menu for the currently displayed image."""
        menu = QMenu(self)

        copy_selected_act = QAction("Copy to Selected", self)
        copy_selected_act.triggered.connect(self.copy_to_selected)
        menu.addAction(copy_selected_act)

        move_selected_act = QAction("Move to Selected", self)
        move_selected_act.triggered.connect(self.move_to_selected)
        menu.addAction(move_selected_act)

        menu.addSeparator()

        start_link_act = QAction("Start Link List", self)
        start_link_act.triggered.connect(self.start_link_list)
        menu.addAction(start_link_act)

        add_link_act = QAction("Add to Link List", self)
        add_link_act.triggered.connect(self.add_to_link_list)
        menu.addAction(add_link_act)

        menu.exec(self.image_label.mapToGlobal(position))

    def _get_selected_dest_folder(self):
        """Get the currently selected destination folder from the main window."""
        main_win = self.parent_app or find_main_window()
        if not main_win:
            return None

        current_index = main_win.dest_tree.currentIndex()
        if current_index.isValid():
            return main_win.dest_model.filePath(
                main_win.dest_proxy.mapToSource(current_index)
            )
        return main_win.dest_root_path

    def copy_to_selected(self):
        """Copy the currently displayed image to the selected destination folder."""
        self._transfer_current_image(is_copy=True)

    def move_to_selected(self):
        """Move the currently displayed image to the selected destination folder.

        Advances to the next image in the current set afterward. If there is
        no next image, shows the "NO IMAGES TO DISPLAY" empty state instead.
        """
        self._transfer_current_image(is_copy=False)

    def _transfer_current_image(self, is_copy):
        if not self.image_list:
            return

        dest_folder = self._get_selected_dest_folder()
        if not dest_folder:
            QMessageBox.warning(
                self, "No Destination", "Please select a destination folder first."
            )
            return

        main_win = self.parent_app or find_main_window()
        if not main_win:
            return

        src_path = self.image_list[self.current_index]

        operation_type = "copy" if is_copy else "move"
        if hasattr(main_win, "start_operation_batch"):
            main_win.start_operation_batch(operation_type)

        try:
            main_win.smart_transfer(src_path, dest_folder, is_copy)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not transfer {os.path.basename(src_path)}:\n{e}",
            )
            return

        operation = "Copied" if is_copy else "Moved"
        if hasattr(main_win, "status_bar"):
            main_win.status_bar.showMessage(
                f"{operation} {os.path.basename(src_path)} to "
                f"{os.path.basename(dest_folder)}",
                3000,
            )

        if is_copy:
            return

        # Move: sync the main window's thumbnail list, then advance to the
        # next image in the current set (without wrapping back to a previous
        # one if there isn't a next image)
        if hasattr(main_win, "remove_thumbnails_by_path"):
            main_win.remove_thumbnails_by_path([src_path])

        self.image_list.pop(self.current_index)
        self.load_image()

    def start_link_list(self):
        """Start a fresh link list seeded with the currently displayed image's path."""
        path = self.image_list[self.current_index]
        reset_link_list(self, path)

    def add_to_link_list(self):
        """Append the currently displayed image's path to the link list."""
        path = self.image_list[self.current_index]
        append_to_link_list(path)

    def eventFilter(self, source, event):
        if event.type() == event.Type.KeyPress and source is self.scroll_area:
            self.keyPressEvent(event)
            return True
        # Update crop dimensions when the overlay changes
        elif source == self.crop_overlay and event.type() in (
            event.Type.MouseMove,
            event.Type.MouseButtonRelease,
            event.Type.Paint,
        ):
            # Use a timer to debounce updates during drag operations
            if not hasattr(self, "_crop_update_timer"):
                self._crop_update_timer = QTimer()
                self._crop_update_timer.setSingleShot(True)
                self._crop_update_timer.timeout.connect(
                    self.update_crop_dimensions_display
                )
            self._crop_update_timer.start(10)
        return super().eventFilter(source, event)

    def handle_external_deletion(self, deleted_paths):
        """Called when the main list deletes files to keep the preview in sync."""
        if not self.image_list:
            return

        # Normalize paths for comparison
        deleted_set = {os.path.normpath(p) for p in deleted_paths}
        current_path = os.path.normpath(self.image_list[self.current_index])

        # 1. Update the internal image_list by removing all deleted paths
        self.image_list = [
            p for p in self.image_list if os.path.normpath(p) not in deleted_set
        ]

        # 2. If the current image was deleted, we need to refresh the view
        if current_path in deleted_set:
            if not self.image_list:
                self.load_image()  # Shows "No images to display"
            else:
                # Adjust index so we don't go out of bounds
                if self.current_index >= len(self.image_list):
                    self.current_index = len(self.image_list) - 1
                self.load_image()
        else:
            # Current image still exists, but its index in the list might have changed
            # Find its new position to keep navigation working correctly
            try:
                self.current_index = self.image_list.index(
                    next(
                        p
                        for p in self.image_list
                        if os.path.normpath(p) == current_path
                    )
                )
                # Update title only (index counts will have changed)
                self.setWindowTitle(
                    f"Preview [{self.current_index + 1}/{len(self.image_list)}]: {os.path.basename(current_path)}"
                )
            except StopIteration:
                pass

    def closeEvent(self, event):
        """Save settings before closing the preview window."""
        main_win = self.parent_app or find_main_window()
        if main_win:
            # Save current geometry to the main window
            main_win.preview_geometry = self.saveGeometry()
            # Persist to disk immediately
            main_win.write_settings()

        event.accept()
