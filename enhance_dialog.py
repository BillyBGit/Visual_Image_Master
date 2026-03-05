"""
Enhancement dialog for Visual Image Master 2026.

Provides a QDialog for selecting enhancement operations, configuring parameters,
live before/after preview (single-image mode), and batch processing (multi-image mode).
"""

import os
import cv2
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QFileDialog,
    QProgressDialog,
    QMessageBox,
    QApplication,
    QSizePolicy,
    QWidget,
    QScrollArea,
)

from image_enhance import (
    OPERATIONS,
    apply_enhancement,
    check_spandrel_available,
    check_cuda_available,
    scan_model_directory,
    get_model_info,
    sharpen,
    denoise,
    adjust_brightness_contrast,
    auto_levels,
    adjust_saturation,
)


def find_main_window():
    """Find the main ImageSorter window."""
    from utils import find_main_window as _find

    return _find()


class PreviewWidget(QLabel):
    """Split-view preview showing original (left) and enhanced (right)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")

        self._original_img = None  # numpy BGR
        self._enhanced_img = None  # numpy BGR
        self._split_pos = 0.5  # 0.0 to 1.0, position of divider
        self._dragging = False

    def set_images(self, original_bgr, enhanced_bgr=None):
        """Set the original and enhanced images for display."""
        self._original_img = original_bgr
        self._enhanced_img = enhanced_bgr
        self._update_display()

    def set_enhanced(self, enhanced_bgr):
        """Update only the enhanced side."""
        self._enhanced_img = enhanced_bgr
        self._update_display()

    def _update_display(self):
        """Render the split-view composite."""
        if self._original_img is None:
            return

        # Scale to fit widget
        widget_w = self.width() - 2  # Account for border
        widget_h = self.height() - 2
        if widget_w <= 0 or widget_h <= 0:
            return

        img_h, img_w = self._original_img.shape[:2]
        scale = min(widget_w / img_w, widget_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        # Resize original
        orig_resized = cv2.resize(
            self._original_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA
        )

        # If we have an enhanced image, create split view
        if self._enhanced_img is not None:
            enh_resized = cv2.resize(
                self._enhanced_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA
            )
            split_x = int(disp_w * self._split_pos)

            # Composite: left = original, right = enhanced
            composite = orig_resized.copy()
            composite[:, split_x:] = enh_resized[:, split_x:]
        else:
            composite = orig_resized
            split_x = int(disp_w * self._split_pos)

        # Convert BGR to RGB for QImage
        rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw split line with handle
        pen = QPen(QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, disp_h)

        # Draw drag handle arrows on the split line
        handle_y = disp_h // 2
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor(40, 40, 40, 200), 1))
        # Handle pill shape
        handle_rect = QRectF(split_x - 12, handle_y - 14, 24, 28)
        painter.drawRoundedRect(handle_rect, 6, 6)
        # Left/right arrows
        painter.setPen(QPen(QColor(40, 40, 40, 230), 2))
        painter.drawLine(split_x - 6, handle_y, split_x - 2, handle_y - 4)
        painter.drawLine(split_x - 6, handle_y, split_x - 2, handle_y + 4)
        painter.drawLine(split_x + 6, handle_y, split_x + 2, handle_y - 4)
        painter.drawLine(split_x + 6, handle_y, split_x + 2, handle_y + 4)

        # Labels with contrasting background pills
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        # "Original" label (top-left, cyan on dark background)
        orig_text = "Original"
        orig_rect = fm.boundingRect(orig_text)
        orig_bg = QRectF(4, 4, orig_rect.width() + 12, orig_rect.height() + 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(orig_bg, 4, 4)
        painter.setPen(QColor(80, 220, 255))  # Cyan
        painter.drawText(10, 4 + fm.ascent() + 3, orig_text)

        # "Enhanced" label (top-right of split, green on dark background)
        if self._enhanced_img is not None:
            enh_text = "Enhanced"
            enh_rect = fm.boundingRect(enh_text)
            enh_bg = QRectF(
                split_x + 4, 4, enh_rect.width() + 12, enh_rect.height() + 6
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(enh_bg, 4, 4)
            painter.setPen(QColor(80, 255, 120))  # Green
            painter.drawText(split_x + 10, 4 + fm.ascent() + 3, enh_text)

        painter.end()
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
            self._update_split_from_mouse(event.position().x())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_split_from_mouse(event.position().x())
        else:
            # Show split cursor when hovering near the divider line
            if self._is_near_divider(event.position().x()):
                self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event):
        self._dragging = False
        if not self._is_near_divider(event.position().x()):
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _is_near_divider(self, mouse_x, threshold=12):
        """Check if mouse is near the split divider line."""
        pixmap = self.pixmap()
        if not pixmap or pixmap.width() <= 0:
            return False
        offset_x = (self.width() - pixmap.width()) / 2
        split_screen_x = offset_x + pixmap.width() * self._split_pos
        return abs(mouse_x - split_screen_x) < threshold

    def _update_split_from_mouse(self, mouse_x):
        pixmap = self.pixmap()
        if pixmap and pixmap.width() > 0:
            # Account for centering
            offset_x = (self.width() - pixmap.width()) / 2
            rel_x = mouse_x - offset_x
            self._split_pos = max(0.05, min(0.95, rel_x / pixmap.width()))
            self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class EnhanceWorker(QThread):
    """Background worker for batch enhancement processing."""

    progress = Signal(int, str)  # (index, status_message)
    finished = Signal(int, int, list)  # (success_count, error_count, error_list)

    def __init__(self, image_paths, operation, params, save_mode, suffix):
        super().__init__()
        self.image_paths = image_paths
        self.operation = operation
        self.params = params
        self.save_mode = save_mode
        self.suffix = suffix
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        success = 0
        errors = []

        for i, path in enumerate(self.image_paths):
            if self._cancelled:
                break

            filename = os.path.basename(path)
            self.progress.emit(i, f"Processing {filename}...")

            # Determine output path
            if self.save_mode == "copy":
                name, ext = os.path.splitext(path)
                output_path = f"{name}{self.suffix}{ext}"
            else:
                output_path = path

            ok, msg = apply_enhancement(path, self.operation, self.params, output_path)
            if ok:
                success += 1
            else:
                errors.append(f"{filename}: {msg}")

        self.finished.emit(success, len(errors), errors)


class EnhanceDialog(QDialog):
    """Dialog for configuring and applying image enhancements."""

    def __init__(self, image_paths, parent=None, preset_operation=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.is_batch = len(image_paths) > 1
        self._worker = None
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._update_preview)

        # For preview: load first image
        self._original_img = None
        self._preview_img = None  # Downsampled for fast preview
        if not self.is_batch and image_paths:
            self._load_preview_image(image_paths[0])

        # Load saved settings
        self._load_settings()

        self._setup_ui()

        # Set preset operation if specified
        if preset_operation:
            idx = (
                self._operation_keys.index(preset_operation)
                if preset_operation in self._operation_keys
                else 0
            )
            self.operation_combo.setCurrentIndex(idx)

        self.setMinimumSize(750, 550)
        self.setWindowTitle(
            "Image Enhance" + (f" ({len(image_paths)} images)" if self.is_batch else "")
        )

        # Initial preview
        if not self.is_batch:
            self._schedule_preview()

    def _load_preview_image(self, path):
        """Load image for preview, downsampled for performance."""
        self._original_img = cv2.imread(path)
        if self._original_img is not None:
            h, w = self._original_img.shape[:2]
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                self._preview_img = cv2.resize(
                    self._original_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            else:
                self._preview_img = self._original_img.copy()

    def _load_settings(self):
        """Load last-used settings from main window."""
        main_win = self.parent() or find_main_window()
        self._save_mode = (
            getattr(main_win, "enhance_save_mode", "overwrite")
            if main_win
            else "overwrite"
        )
        self._copy_suffix = (
            getattr(main_win, "enhance_copy_suffix", "_enhanced")
            if main_win
            else "_enhanced"
        )
        self._last_operation = (
            getattr(main_win, "enhance_last_operation", "sharpen")
            if main_win
            else "sharpen"
        )
        self._last_params = (
            getattr(main_win, "enhance_last_params", {}) if main_win else {}
        )
        self._model_dir = getattr(main_win, "enhance_model_dir", "") if main_win else ""

    def _save_settings(self):
        """Save current settings to main window."""
        main_win = self.parent() or find_main_window()
        if not main_win:
            return

        main_win.enhance_save_mode = self._save_mode
        main_win.enhance_copy_suffix = self._copy_suffix
        main_win.enhance_last_operation = self._current_operation_key()
        main_win.enhance_last_params = self._last_params
        main_win.enhance_model_dir = self._model_dir

        if hasattr(main_win, "_settings_loaded") and main_win._settings_loaded:
            main_win.write_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Operation selector ---
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Operation:"))
        self.operation_combo = QComboBox()

        self._operation_keys = []
        self._spandrel_available = check_spandrel_available()

        for key, op in OPERATIONS.items():
            self._operation_keys.append(key)
            self.operation_combo.addItem(op["name"])

        # Set to last used operation
        if self._last_operation in self._operation_keys:
            self.operation_combo.setCurrentIndex(
                self._operation_keys.index(self._last_operation)
            )

        self.operation_combo.currentIndexChanged.connect(self._on_operation_changed)
        op_layout.addWidget(self.operation_combo, 1)
        layout.addLayout(op_layout)

        # --- Main content: params + preview ---
        content_layout = QHBoxLayout()

        # Left side: parameters (inside scroll area for overflow)
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout()
        self.params_layout.setVerticalSpacing(10)
        self.params_layout.setContentsMargins(8, 12, 8, 12)
        self.params_group.setLayout(self.params_layout)
        self.params_group.setMinimumWidth(280)
        self.params_group.setMaximumWidth(350)

        params_scroll = QScrollArea()
        params_scroll.setWidget(self.params_group)
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        params_scroll.setMinimumWidth(280)
        params_scroll.setMaximumWidth(350)
        content_layout.addWidget(params_scroll)

        # Right side: preview (single-image mode only)
        if not self.is_batch:
            self.preview_widget = PreviewWidget()
            content_layout.addWidget(self.preview_widget, 1)
        else:
            # In batch mode, show info label
            batch_label = QLabel(
                f"Batch mode: {len(self.image_paths)} images will be processed.\n\n"
                f"Adjust parameters, then click 'Apply to All'."
            )
            batch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            batch_label.setStyleSheet("color: #aaa; font-size: 14px; padding: 20px;")
            content_layout.addWidget(batch_label, 1)

        layout.addLayout(content_layout)

        # --- AI Upscale panel (hidden by default) ---
        self.ai_panel = self._create_ai_panel()
        layout.addWidget(self.ai_panel)
        self.ai_panel.setVisible(False)

        # --- Save mode ---
        save_group = QGroupBox("Save Mode")
        save_layout = QHBoxLayout()

        self.save_mode_group = QButtonGroup(self)
        self.overwrite_radio = QRadioButton("Overwrite original")
        self.copy_radio = QRadioButton("Save as copy")
        self.save_mode_group.addButton(self.overwrite_radio)
        self.save_mode_group.addButton(self.copy_radio)

        if self._save_mode == "copy":
            self.copy_radio.setChecked(True)
        else:
            self.overwrite_radio.setChecked(True)

        save_layout.addWidget(self.overwrite_radio)
        save_layout.addWidget(self.copy_radio)

        save_layout.addWidget(QLabel("Suffix:"))
        self.suffix_edit = QLineEdit(self._copy_suffix)
        self.suffix_edit.setMaximumWidth(120)
        save_layout.addWidget(self.suffix_edit)
        save_layout.addStretch()

        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        # --- Action buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        if not self.is_batch:
            self.preview_btn = QPushButton("Preview")
            self.preview_btn.clicked.connect(self._update_preview)
            button_layout.addWidget(self.preview_btn)

        self.apply_btn = QPushButton("Apply" if not self.is_batch else "Apply to All")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(self.apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        # Tracker for parameter widgets
        self._param_widgets = {}

        # Build initial parameter panel
        self._on_operation_changed(self.operation_combo.currentIndex())

    def _create_ai_panel(self):
        """Create the AI Model configuration panel."""
        panel = QGroupBox("AI Model Settings")
        layout = QFormLayout()

        # Default model directory to project's models/ folder if not set
        if not self._model_dir:
            default_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models"
            )
            if os.path.isdir(default_dir):
                self._model_dir = default_dir

        # Warning if spandrel not installed
        if not check_spandrel_available():
            warn_label = QLabel(
                "Install required: <code>pip install spandrel torch</code>"
            )
            warn_label.setTextFormat(Qt.TextFormat.RichText)
            warn_label.setStyleSheet(
                "color: #e88; font-weight: bold; padding: 4px; "
                "border: 1px solid #c66; border-radius: 3px; background: #332;"
            )
            layout.addRow(warn_label)

        # Model directory
        dir_layout = QHBoxLayout()
        self.model_dir_edit = QLineEdit(self._model_dir)
        self.model_dir_edit.setPlaceholderText("Path to model files directory...")
        dir_layout.addWidget(self.model_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_model_dir)
        dir_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_model_list)
        dir_layout.addWidget(refresh_btn)

        layout.addRow("Model Directory:", dir_layout)

        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        layout.addRow("Model:", self.model_combo)

        # Model info
        self.model_info_label = QLabel("Select a model to see details")
        self.model_info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addRow("Info:", self.model_info_label)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        if check_cuda_available():
            self.device_combo.addItem("CUDA (GPU)", "cuda")
            self.device_combo.setCurrentIndex(1)
        layout.addRow("Device:", self.device_combo)

        # Tile size
        self.tile_size_combo = QComboBox()
        for size in [0, 128, 256, 512, 1024]:
            label = "No tiling (whole image)" if size == 0 else f"{size}x{size}"
            self.tile_size_combo.addItem(label, size)
        self.tile_size_combo.setCurrentIndex(3)  # Default 512
        layout.addRow("Tile Size:", self.tile_size_combo)

        panel.setLayout(layout)

        # Populate model list if directory already set
        if self._model_dir:
            self._refresh_model_list()

        return panel

    def _browse_model_dir(self):
        """Open directory browser for model files."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", self._model_dir or ""
        )
        if dir_path:
            self.model_dir_edit.setText(dir_path)
            self._model_dir = dir_path
            self._refresh_model_list()

    def _refresh_model_list(self):
        """Scan model directory and populate combo box."""
        self.model_combo.clear()
        dir_path = self.model_dir_edit.text().strip()
        if not dir_path:
            return

        self._model_dir = dir_path
        models = scan_model_directory(dir_path)

        if not models:
            self.model_combo.addItem("No model files found")
            return

        for m in models:
            self.model_combo.addItem(f"{m['name']} ({m['size_mb']} MB)", m["path"])

    def _on_model_selected(self, index):
        """Show info about the selected model."""
        model_path = self.model_combo.currentData()
        if not model_path or not os.path.isfile(str(model_path)):
            self.model_info_label.setText("Select a model to see details")
            return

        self.model_info_label.setText("Loading model info...")
        QApplication.processEvents()

        device = self.device_combo.currentData() or "cpu"
        info = get_model_info(model_path, device)

        if info:
            self.model_info_label.setText(
                f"Architecture: {info['architecture']} | Scale: {info['scale']}x | "
                f"Channels: {info['input_channels']}→{info['output_channels']}"
            )
            self.model_info_label.setStyleSheet("color: #4a9; font-weight: bold;")
        else:
            self.model_info_label.setText("Could not load model info")
            self.model_info_label.setStyleSheet("color: #c44; font-style: italic;")

    def _current_operation_key(self):
        """Get the key of the currently selected operation."""
        idx = self.operation_combo.currentIndex()
        if 0 <= idx < len(self._operation_keys):
            return self._operation_keys[idx]
        return "sharpen"

    def _on_operation_changed(self, index):
        """Rebuild parameter panel when operation changes."""
        op_key = self._current_operation_key()
        op_meta = OPERATIONS.get(op_key, {})

        # Show/hide AI panel
        is_ai = op_key == "ai_upscale"
        self.ai_panel.setVisible(is_ai)

        # Clear existing param widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets.clear()

        # Skip param panel for AI model (handled by ai_panel) and resize (custom)
        if is_ai:
            if not self._spandrel_available:
                install_label = QLabel(
                    "<b>Requires additional packages</b><br><br>"
                    "To use AI models (.pth / .safetensors),<br>"
                    "install spandrel and PyTorch:<br><br>"
                    "<code>pip install spandrel torch</code><br><br>"
                    "Then restart the application.<br><br>"
                    "Supports 50+ architectures including<br>"
                    "ESRGAN, SwinIR, HAT, SPAN, SCUNet,<br>"
                    "GFPGAN, CodeFormer, and more."
                )
                install_label.setTextFormat(Qt.TextFormat.RichText)
                install_label.setWordWrap(True)
                install_label.setStyleSheet("color: #e88; padding: 8px;")
                self.params_layout.addRow(install_label)
            else:
                info_label = QLabel("Configure AI model settings below.")
                info_label.setStyleSheet("color: #888;")
                self.params_layout.addRow(info_label)
            return

        if op_key == "resize":
            self._build_resize_params()
            return

        # Build sliders for standard operations
        params = op_meta.get("params", [])
        saved_params = self._last_params.get(op_key, {})

        for p in params:
            key = p["key"]
            label = p["label"]
            p_type = p["type"]
            default = saved_params.get(key, p["default"])

            if p_type == "float":
                self._add_float_param(
                    key, label, p["min"], p["max"], default, p["step"]
                )
            elif p_type == "int":
                self._add_int_param(key, label, p["min"], p["max"], default, p["step"])
            elif p_type == "bool":
                self._add_bool_param(key, label, default)

        if not self.is_batch:
            self._schedule_preview()

    def _build_resize_params(self):
        """Build the resize-specific parameter panel."""
        saved = self._last_params.get("resize", {})

        # Get current image dimensions for defaults
        img_w, img_h = 1024, 1024
        if self._original_img is not None:
            img_h, img_w = self._original_img.shape[:2]

        # Mode toggle
        self.resize_mode_check = QCheckBox("Scale by percentage")
        self.resize_mode_check.setChecked(saved.get("use_scale", False))
        self.resize_mode_check.toggled.connect(self._on_resize_mode_changed)
        self.params_layout.addRow(self.resize_mode_check)

        # Width/Height
        self._add_int_param("width", "Width", 1, 16384, saved.get("width", img_w), 1)
        self._add_int_param("height", "Height", 1, 16384, saved.get("height", img_h), 1)

        # Aspect ratio
        self._add_bool_param(
            "maintain_aspect",
            "Maintain Aspect Ratio",
            saved.get("maintain_aspect", True),
        )

        # Scale percentage
        self._add_int_param(
            "scale_percent", "Scale %", 1, 1000, saved.get("scale_percent", 100), 5
        )

        self._on_resize_mode_changed(self.resize_mode_check.isChecked())

    def _on_resize_mode_changed(self, use_scale):
        """Toggle between dimension and percentage resize modes."""
        for key in ("width", "height", "maintain_aspect"):
            if key in self._param_widgets:
                for w in self._param_widgets[key]:
                    w.setEnabled(not use_scale)

        if "scale_percent" in self._param_widgets:
            for w in self._param_widgets["scale_percent"]:
                w.setEnabled(use_scale)

        if not self.is_batch:
            self._schedule_preview()

    def _add_float_param(self, key, label, min_val, max_val, default, step):
        """Add a float parameter with slider + spinbox."""
        container = QWidget()
        container.setMinimumHeight(32)
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 4, 0, 4)

        # Slider (integer internally, mapped to float)
        steps = int((max_val - min_val) / step)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(steps)
        slider.setValue(int((default - min_val) / step))

        # SpinBox
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(2)
        spinbox.setValue(default)
        spinbox.setFixedWidth(80)

        # Sync slider <-> spinbox
        def on_slider(val, _min=min_val, _step=step, _sb=spinbox):
            _sb.blockSignals(True)
            _sb.setValue(_min + val * _step)
            _sb.blockSignals(False)
            if not self.is_batch:
                self._schedule_preview()

        def on_spinbox(val, _min=min_val, _step=step, _sl=slider):
            _sl.blockSignals(True)
            _sl.setValue(int((val - _min) / _step))
            _sl.blockSignals(False)
            if not self.is_batch:
                self._schedule_preview()

        slider.valueChanged.connect(on_slider)
        spinbox.valueChanged.connect(on_spinbox)

        h_layout.addWidget(slider, 1)
        h_layout.addWidget(spinbox)

        self.params_layout.addRow(f"{label}:", container)
        self._param_widgets[key] = [container, slider, spinbox]

    def _add_int_param(self, key, label, min_val, max_val, default, step):
        """Add an integer parameter with slider + spinbox."""
        container = QWidget()
        container.setMinimumHeight(32)
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 4, 0, 4)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setSingleStep(step)
        slider.setValue(default)

        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(default)
        spinbox.setFixedWidth(80)

        def on_slider(val, _sb=spinbox):
            _sb.blockSignals(True)
            _sb.setValue(val)
            _sb.blockSignals(False)
            if not self.is_batch:
                self._schedule_preview()

        def on_spinbox(val, _sl=slider):
            _sl.blockSignals(True)
            _sl.setValue(val)
            _sl.blockSignals(False)
            if not self.is_batch:
                self._schedule_preview()

        slider.valueChanged.connect(on_slider)
        spinbox.valueChanged.connect(on_spinbox)

        h_layout.addWidget(slider, 1)
        h_layout.addWidget(spinbox)

        self.params_layout.addRow(f"{label}:", container)
        self._param_widgets[key] = [container, slider, spinbox]

    def _add_bool_param(self, key, label, default):
        """Add a boolean parameter with checkbox."""
        checkbox = QCheckBox()
        checkbox.setChecked(default)
        checkbox.toggled.connect(
            lambda: self._schedule_preview() if not self.is_batch else None
        )

        self.params_layout.addRow(f"{label}:", checkbox)
        self._param_widgets[key] = [checkbox]

    def _get_current_params(self):
        """Collect current parameter values from UI widgets."""
        op_key = self._current_operation_key()
        params = {}

        if op_key == "ai_upscale":
            params["model_path"] = self.model_combo.currentData() or ""
            params["device"] = self.device_combo.currentData() or "cpu"
            params["tile_size"] = self.tile_size_combo.currentData() or 512
            return params

        op_meta = OPERATIONS.get(op_key, {})
        param_defs = op_meta.get("params", [])

        for p in param_defs:
            key = p["key"]
            if key not in self._param_widgets:
                params[key] = p["default"]
                continue

            widgets = self._param_widgets[key]
            p_type = p["type"]

            if p_type == "float":
                # SpinBox is the last widget
                params[key] = widgets[-1].value()
            elif p_type == "int":
                params[key] = widgets[-1].value()
            elif p_type == "bool":
                params[key] = widgets[0].isChecked()

        # Handle resize special case
        if op_key == "resize":
            if hasattr(self, "resize_mode_check"):
                params["use_scale"] = self.resize_mode_check.isChecked()

        return params

    def _schedule_preview(self):
        """Schedule a debounced preview update."""
        if not self.is_batch:
            self._preview_timer.start()

    def _update_preview(self):
        """Apply current operation to preview image and display."""
        if self.is_batch or self._preview_img is None:
            return
        if not hasattr(self, "preview_widget"):
            return

        op_key = self._current_operation_key()
        params = self._get_current_params()

        # Skip AI upscale in preview (too slow)
        if op_key == "ai_upscale":
            self.preview_widget.set_images(self._preview_img, None)
            return

        try:
            if op_key == "resize":
                # For resize preview, just show original with dimensions info
                self.preview_widget.set_images(self._preview_img, None)
                return

            # Apply operation to preview image
            if op_key == "sharpen":
                enhanced = sharpen(
                    self._preview_img,
                    amount=params.get("amount", 1.0),
                    radius=params.get("radius", 2.0),
                    threshold=params.get("threshold", 0),
                )
            elif op_key == "denoise":
                enhanced = denoise(
                    self._preview_img,
                    strength=params.get("strength", 10),
                    color_strength=params.get("color_strength", 10),
                )
            elif op_key == "brightness_contrast":
                enhanced = adjust_brightness_contrast(
                    self._preview_img,
                    brightness=params.get("brightness", 0),
                    contrast=params.get("contrast", 1.0),
                )
            elif op_key == "auto_levels":
                enhanced = auto_levels(
                    self._preview_img,
                    clip_limit=params.get("clip_limit", 2.0),
                    grid_size=params.get("grid_size", 8),
                )
            elif op_key == "saturation":
                enhanced = adjust_saturation(
                    self._preview_img, factor=params.get("factor", 1.0)
                )
            else:
                enhanced = self._preview_img.copy()

            self.preview_widget.set_images(self._preview_img, enhanced)

        except Exception as e:
            debug = os.environ.get("IMG_SORT_DEBUG") == "1"
            if debug:
                print(f"[DEBUG] Preview error: {e}")

    def _on_apply(self):
        """Apply enhancement to image(s)."""
        op_key = self._current_operation_key()
        params = self._get_current_params()

        # Save current params
        self._last_params[op_key] = params
        self._save_mode = "copy" if self.copy_radio.isChecked() else "overwrite"
        self._copy_suffix = self.suffix_edit.text() or "_enhanced"

        if self.is_batch:
            self._apply_batch(op_key, params)
        else:
            self._apply_single(op_key, params)

    def _apply_single(self, op_key, params):
        """Apply enhancement to a single image."""
        path = self.image_paths[0]

        # Determine output path
        if self._save_mode == "copy":
            name, ext = os.path.splitext(path)
            output_path = f"{name}{self._copy_suffix}{ext}"
        else:
            output_path = path

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        ok, msg = apply_enhancement(path, op_key, params, output_path)
        QApplication.restoreOverrideCursor()

        if ok:
            self._save_settings()
            self.accept()
        else:
            QMessageBox.warning(self, "Enhancement Failed", msg)

    def _apply_batch(self, op_key, params):
        """Apply enhancement to multiple images with progress."""
        suffix = self._copy_suffix if self._save_mode == "copy" else ""

        progress = QProgressDialog(
            "Applying enhancement...", "Cancel", 0, len(self.image_paths), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        self._worker = EnhanceWorker(
            self.image_paths, op_key, params, self._save_mode, suffix
        )

        def on_progress(idx, msg):
            progress.setValue(idx)
            progress.setLabelText(msg)
            QApplication.processEvents()

        def on_finished(success, error_count, errors):
            progress.close()
            self._worker = None

            summary = f"Enhanced {success} of {len(self.image_paths)} images."
            if error_count > 0:
                summary += f"\n\n{error_count} error(s):\n" + "\n".join(errors[:20])
                if error_count > 20:
                    summary += f"\n... and {error_count - 20} more"
                QMessageBox.warning(self, "Enhancement Complete", summary)
            else:
                QMessageBox.information(self, "Enhancement Complete", summary)

            self._save_settings()
            self.accept()

        self._worker.progress.connect(on_progress)
        self._worker.finished.connect(on_finished)
        progress.canceled.connect(self._worker.cancel)

        self._worker.start()

    def closeEvent(self, event):
        """Clean up worker on close."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
