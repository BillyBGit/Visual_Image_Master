"""
Image preview window for viewing and navigating images.
"""

import os
from PySide6.QtWidgets import (QMainWindow, QScrollArea, QLabel, QMessageBox, QApplication,
                                QStatusBar, QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
                                QPushButton, QDialogButtonBox, QFileDialog)
from PySide6.QtCore import Qt, QTimer, QSize, QRect
from PySide6.QtGui import QAction, QPixmap, QActionGroup, QIcon, QIntValidator
from PIL import Image

from utils import find_main_window
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
        self.fit_to_window = getattr(self.parent_app, 'preview_fit_default', False) if self.parent_app else False

        # Crop state
        self.crop_mode = None
        self.crop_overlay = None

        # Extract crop setting
        self.autoname_extract = getattr(self.parent_app, 'preview_autoname_extract', False) if self.parent_app else False

        # 4. UI Setup
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)

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

        self.setStatusBar(self.status_bar)

        # 5. Setup Menu and Sync Checkbox
        self.setup_menu()
        if hasattr(self, 'fit_act'):
            self.fit_act.setChecked(self.fit_to_window)
        if hasattr(self, 'native_act'):
            self.native_act.setChecked(not self.fit_to_window)

        # 6. Handle Window Geometry BEFORE loading image
        saved_geo = getattr(self.parent_app, 'preview_geometry', None) if self.parent_app else None
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

        # Fit to Window Action
        self.fit_act = QAction("Fit to Window", self, checkable=True)
        self.fit_act.triggered.connect(self.toggle_fit_mode)
        options_menu.addAction(self.fit_act)

        # Native Size Action
        self.native_act = QAction("Native Size", self, checkable=True)
        self.native_act.triggered.connect(self.set_native_size)
        options_menu.addAction(self.native_act)

        options_menu.addSeparator()

        # Autoname Extract Action
        self.autoname_extract_act = QAction("Autoname Extract", self, checkable=True)
        self.autoname_extract_act.setChecked(self.autoname_extract)
        self.autoname_extract_act.triggered.connect(self.toggle_autoname_extract)
        options_menu.addAction(self.autoname_extract_act)

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

        crop_menu.addSeparator()

        # Fixed size crops
        for size in ["1024x1024", "768x768", "640x640", "512x512"]:
            act = QAction(size, self, checkable=True)
            act.triggered.connect(lambda checked, s=size: self.set_crop_mode(s))
            self.crop_action_group.addAction(act)
            crop_menu.addAction(act)

        crop_menu.addSeparator()

        # Clear crop
        clear_act = QAction("Clear Crop", self)
        clear_act.triggered.connect(self.clear_crop)
        crop_menu.addAction(clear_act)

    def toggle_fit_mode(self, checked):
        self.fit_to_window = checked
        if hasattr(self, 'native_act'):
            self.native_act.setChecked(not checked)
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
        if hasattr(self, 'fit_act'):
            self.fit_act.setChecked(False)
        if hasattr(self, 'native_act'):
            self.native_act.setChecked(True)
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

        # Disable fit-to-window when cropping
        if mode and self.fit_to_window:
            self.fit_to_window = False
            self.zoom_factor = 1.0
            if hasattr(self, 'fit_act'):
                self.fit_act.setChecked(False)
            if hasattr(self, 'native_act'):
                self.native_act.setChecked(True)
            self.update_image()

        self.update_crop_overlay()

    def clear_crop(self):
        """Clear crop selection."""
        self.crop_mode = None

        # Uncheck all crop actions
        if hasattr(self, 'crop_action_group'):
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
        self.crop_overlay.set_crop_mode(self.crop_mode, image_size)
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
        self.crop_dimensions_label.setText(f"Crop: {orig_width} × {orig_height} (click to edit)")

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
        info_label = QLabel(f"Max dimensions: {self.original_pixmap.width()} × {self.original_pixmap.height()}")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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
                    QMessageBox.warning(self, "Invalid Dimensions", "Width and height must be at least 1 pixel.")
                    return

                if new_width > self.original_pixmap.width() or new_height > self.original_pixmap.height():
                    QMessageBox.warning(self, "Invalid Dimensions",
                        f"Dimensions cannot exceed image size ({self.original_pixmap.width()} × {self.original_pixmap.height()}).")
                    return

                # Convert to displayed coordinates
                new_display_width = int(new_width / scale_x)
                new_display_height = int(new_height / scale_y)

                # Keep top-left corner fixed, adjust bottom-right
                new_rect = QRect(
                    crop_rect.x(),
                    crop_rect.y(),
                    new_display_width,
                    new_display_height
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
                QMessageBox.warning(self, "Invalid Input", "Please enter valid integer dimensions.")
                return

    def load_image(self):
        if not self.image_list:
            self.image_label.setText("No images to display")
            self.image_label.setPixmap(QPixmap())
            self.setWindowTitle("Preview: Empty")
            self.update_zoom_display()
            return

        path = self.image_list[self.current_index]
        self.setWindowTitle(f"Preview [{self.current_index + 1}/{len(self.image_list)}]: {os.path.basename(path)}")
        self.original_pixmap = QPixmap(path)

        # Reset zoom when loading new image in non-fit mode
        if not self.fit_to_window:
            self.zoom_factor = 1.0

        # Clear crop when loading new image
        self.clear_crop()

        self.update_image()
        self.update_zoom_display()

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
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
                self.image_label.adjustSize()
        else:
            # Use zoom factor for manual scaling
            new_width = int(self.original_pixmap.width() * self.zoom_factor)
            if new_width > 0:
                scaled = self.original_pixmap.scaledToWidth(new_width, Qt.TransformationMode.SmoothTransformation)
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
        if main_win and hasattr(main_win, 'conf_delete_act'):
            should_confirm = main_win.conf_delete_act.isChecked()

        do_delete = True
        if should_confirm:
            delete_type = "permanently delete" if permanent else "delete"
            reply = QMessageBox.question(
                self, "Delete Image",
                f"Are you sure you want to {delete_type} {os.path.basename(path_to_delete)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            do_delete = (reply == QMessageBox.StandardButton.Yes)

        if do_delete:
            try:
                # Start a new deletion batch for undo (clears previous undo state)
                if not permanent and main_win and hasattr(main_win, 'delete_undo_manager'):
                    main_win.delete_undo_manager.start_new_deletion()

                # 1. Physical Delete or store for undo
                if os.path.exists(path_to_delete):
                    if permanent:
                        os.remove(path_to_delete)
                    else:
                        # Store for undo (moves to temp location)
                        if main_win and hasattr(main_win, 'delete_undo_manager'):
                            main_win.delete_undo_manager.store_for_undo(path_to_delete)
                        else:
                            os.remove(path_to_delete)

                # 2. UI SYNC: Update the main window's thumbnail list
                if main_win and hasattr(main_win, 'remove_thumbnails_by_path'):
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
                if main_win and hasattr(main_win, 'status_bar'):
                    if permanent:
                        main_win.status_bar.showMessage("Permanently deleted image", 3000)
                    else:
                        main_win.status_bar.showMessage("Deleted image - Ctrl+Z to undo", 3000)

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
            if img_format in ['JPEG', 'JPG']:
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
                self.crop_overlay.crop_ratios = (new_x_ratio, new_y_ratio, new_w_ratio, new_h_ratio)

            # Reload the image in preview
            self.load_image()

            # Update main window thumbnail if accessible
            main_win = self.parent_app or find_main_window()
            if main_win and hasattr(main_win, 'thumb_list'):
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
            if main_win and hasattr(main_win, 'status_bar'):
                main_win.status_bar.showMessage("Rotated image 90° clockwise", 2000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not rotate image:\n{e}")

    def generate_extract_filename(self, original_path):
        """Generate a unique filename for extracted crop with _e0, _e1, etc. suffix."""
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
        auto_filename = self.generate_extract_filename(image_path)

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
                "Images (*.png *.jpg *.jpeg *.gif *.webp *.avif *.tiff *.bmp);;All Files (*)"
            )

            if not extract_path:
                # User cancelled
                return

        try:
            # Load original image with PIL
            with Image.open(image_path) as img:
                # Crop the image
                cropped = img.crop((orig_x, orig_y, orig_x + orig_width, orig_y + orig_height))

                # Save to new file
                # Determine format from extension
                save_format = None
                ext = os.path.splitext(extract_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    save_format = 'JPEG'
                    cropped.save(extract_path, format=save_format, quality=95, optimize=True)
                elif ext == '.png':
                    save_format = 'PNG'
                    cropped.save(extract_path, format=save_format)
                else:
                    # Let PIL auto-detect format
                    cropped.save(extract_path)

            # Show success message in preview window's own status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(
                    f"Extracted crop to: {os.path.basename(extract_path)}",
                    5000
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
        if main_win and hasattr(main_win, 'conf_delete_act'):
            should_confirm = main_win.conf_delete_act.isChecked()

        do_save = True
        if should_confirm:
            reply = QMessageBox.question(
                self, "Confirm Crop",
                f"This will overwrite the original file:\n{os.path.basename(image_path)}\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            do_save = (reply == QMessageBox.StandardButton.Yes)

        if do_save:
            try:
                # Load original image with PIL
                with Image.open(image_path) as img:
                    # Crop the image
                    cropped = img.crop((orig_x, orig_y, orig_x + orig_width, orig_y + orig_height))

                    # Save back to original file
                    cropped.save(image_path)

                # Reload the image
                self.original_pixmap = QPixmap(image_path)
                self.clear_crop()
                self.update_image()

                # Show success message
                if main_win:
                    main_win.status_bar.showMessage(f"Cropped and saved: {os.path.basename(image_path)}", 3000)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save cropped image:\n{e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape or (event.key() == Qt.Key.Key_W and event.modifiers() == Qt.KeyboardModifier.ControlModifier):
            self.close()
        elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+S to save crop
            if self.crop_mode:
                self.save_crop()
        elif event.key() == Qt.Key.Key_E and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+E to extract crop
            if self.crop_mode:
                self.extract_crop()
        elif event.key() == Qt.Key.Key_R and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
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
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """Handle preview window resize - triggers image update for fit mode."""
        super().resizeEvent(event)
        if hasattr(self, 'fit_to_window') and self.fit_to_window:
            # Use a timer to debounce resize events
            if not hasattr(self, '_resize_timer'):
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
                if hasattr(self, 'fit_act'):
                    self.fit_act.setChecked(False)
                if hasattr(self, 'native_act'):
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

    def eventFilter(self, source, event):
        if event.type() == event.Type.KeyPress and source is self.scroll_area:
            self.keyPressEvent(event)
            return True
        # Update crop dimensions when the overlay changes
        elif source == self.crop_overlay and event.type() in (event.Type.MouseMove, event.Type.MouseButtonRelease, event.Type.Paint):
            # Use a timer to debounce updates during drag operations
            if not hasattr(self, '_crop_update_timer'):
                self._crop_update_timer = QTimer()
                self._crop_update_timer.setSingleShot(True)
                self._crop_update_timer.timeout.connect(self.update_crop_dimensions_display)
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
        self.image_list = [p for p in self.image_list if os.path.normpath(p) not in deleted_set]

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
                self.current_index = self.image_list.index(next(
                    p for p in self.image_list if os.path.normpath(p) == current_path
                ))
                # Update title only (index counts will have changed)
                self.setWindowTitle(f"Preview [{self.current_index + 1}/{len(self.image_list)}]: {os.path.basename(current_path)}")
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
