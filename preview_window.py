"""
Image preview window for viewing and navigating images.
"""

import os
from PySide6.QtWidgets import QMainWindow, QScrollArea, QLabel, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import QApplication

from utils import find_main_window


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

        # 4. UI Setup
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)

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

    def toggle_fit_mode(self, checked):
        self.fit_to_window = checked
        if hasattr(self, 'native_act'):
            self.native_act.setChecked(not checked)
        self.update_image()

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

        # Update the main window's setting
        main_win = self.parent_app or find_main_window()
        if main_win:
            main_win.preview_fit_default = False
            main_win.preview_geometry = self.saveGeometry()
            main_win.write_settings()

    def load_image(self):
        if not self.image_list:
            self.image_label.setText("No images to display")
            self.image_label.setPixmap(QPixmap())
            self.setWindowTitle("Preview: Empty")
            return

        path = self.image_list[self.current_index]
        self.setWindowTitle(f"Preview [{self.current_index + 1}/{len(self.image_list)}]: {os.path.basename(path)}")
        self.original_pixmap = QPixmap(path)

        # Reset zoom when loading new image in non-fit mode
        if not self.fit_to_window:
            self.zoom_factor = 1.0

        self.update_image()

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

    def delete_current_image(self):
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
            reply = QMessageBox.question(
                self, "Delete Image",
                f"Are you sure you want to delete {os.path.basename(path_to_delete)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            do_delete = (reply == QMessageBox.StandardButton.Yes)

        if do_delete:
            try:
                # 1. Physical Delete
                if os.path.exists(path_to_delete):
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

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape or (event.key() == Qt.Key.Key_W and event.modifiers() == Qt.KeyboardModifier.ControlModifier):
            self.close()
        elif event.key() == Qt.Key.Key_Left:
            if self.image_list:
                self.current_index = (self.current_index - 1) % len(self.image_list)
                self.load_image()
        elif event.key() == Qt.Key.Key_Right:
            if self.image_list:
                self.current_index = (self.current_index + 1) % len(self.image_list)
                self.load_image()
        elif event.key() == Qt.Key.Key_Delete:
            self.delete_current_image()
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
            if not self.fit_to_window:
                delta = event.angleDelta().y()
                self.zoom_factor *= 1.1 if delta > 0 else 0.9
                self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
                self.update_image()
                event.accept()
        else:
            # Pass to scroll area for normal scrolling
            super().wheelEvent(event)

    def eventFilter(self, source, event):
        if event.type() == event.Type.KeyPress and source is self.scroll_area:
            self.keyPressEvent(event)
            return True
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
