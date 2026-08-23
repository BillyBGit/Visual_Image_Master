"""
Auto Caption window for generating and editing image captions.
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
    QTextEdit,
    QLineEdit,
    QMessageBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QPixmap, QKeySequence, QShortcut

from utils import find_main_window
from caption_generator import get_caption_generator, CaptionWorker, BlendWorker


class CaptionWindow(QMainWindow):
    """Window for generating and editing image captions."""

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
        self.displayed_pixmap = QPixmap()

        # View settings
        self.fit_to_window = True  # Start in fit mode for caption viewing

        # Caption state
        self.has_unsaved_changes = False
        self.current_caption_path = None
        self.caption_worker = None
        self.blend_worker = None

        # Undo/redo history for generated caption (stores previous states)
        self._generated_undo_stack = []  # Stack of previous texts
        self._generated_redo_stack = []  # Stack for redo

        # Initialization flag - prevents re-initialization on subsequent show events
        self._initialized = False
        self._maximize_attempts = 0

        # Caption generator (singleton)
        self.generator = get_caption_generator()

        # Set window title
        self.setWindowTitle("Auto Caption Tool")

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._load_settings()

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

        # Right side: Caption panel
        self.right_panel = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        self.right_panel_layout.setContentsMargins(10, 10, 10, 10)
        self.right_panel_layout.setSpacing(10)

        # Backend selection group
        backend_group = QGroupBox("Caption Backend")
        backend_layout = QVBoxLayout(backend_group)

        # Backend dropdown
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Florence2 (Local)", "florence2")
        self.backend_combo.addItem("WD14 Tagger (Local)", "wd14")
        self.backend_combo.addItem("Ollama (Server)", "ollama")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        backend_row.addWidget(self.backend_combo, 1)
        backend_layout.addLayout(backend_row)

        # Status label (used by Florence2 and WD14)
        self.backend_status = QLabel()
        self.backend_status.setWordWrap(True)
        self.backend_status.setStyleSheet("QLabel { color: gray; font-size: 11px; }")
        backend_layout.addWidget(self.backend_status)

        # Florence2 settings widget
        self.florence_widget = QWidget()
        florence_layout = QVBoxLayout(self.florence_widget)
        florence_layout.setContentsMargins(0, 5, 0, 0)

        # Model size dropdown with sizes
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.florence_model_combo = QComboBox()
        self.florence_model_combo.addItem("Base (~460MB)", "base")
        self.florence_model_combo.addItem("Large (~1.5GB)", "large")
        self.florence_model_combo.addItem("Base Fine-tuned (~460MB)", "base-ft")
        self.florence_model_combo.addItem("Large Fine-tuned (~1.5GB)", "large-ft")
        self.florence_model_combo.addItem("Custom Path...", "custom")
        self.florence_model_combo.currentIndexChanged.connect(
            self._on_florence_model_changed
        )
        model_row.addWidget(self.florence_model_combo, 1)
        florence_layout.addLayout(model_row)

        # Custom model path (hidden by default)
        self.custom_path_widget = QWidget()
        custom_path_layout = QHBoxLayout(self.custom_path_widget)
        custom_path_layout.setContentsMargins(0, 0, 0, 0)
        custom_path_layout.addWidget(QLabel("Path:"))
        self.florence_custom_path = QLineEdit()
        self.florence_custom_path.setPlaceholderText("HuggingFace ID or local path")
        custom_path_layout.addWidget(self.florence_custom_path, 1)
        florence_layout.addWidget(self.custom_path_widget)
        self.custom_path_widget.hide()

        # Detail level dropdown
        detail_row = QHBoxLayout()
        detail_row.addWidget(QLabel("Detail:"))
        self.florence_detail_combo = QComboBox()
        self.florence_detail_combo.addItem("Brief (short caption)", "brief")
        self.florence_detail_combo.addItem("Standard", "standard")
        self.florence_detail_combo.addItem("Detailed (comprehensive)", "detailed")
        self.florence_detail_combo.setCurrentIndex(2)  # Default to detailed
        detail_row.addWidget(self.florence_detail_combo, 1)
        florence_layout.addLayout(detail_row)

        backend_layout.addWidget(self.florence_widget)

        # Ollama settings (initially hidden)
        self.ollama_widget = QWidget()
        ollama_layout = QVBoxLayout(self.ollama_widget)
        ollama_layout.setContentsMargins(0, 5, 0, 0)

        # Ollama URL
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self.ollama_url = QLineEdit()
        self.ollama_url.setPlaceholderText("http://localhost:11434")
        self.ollama_url.setText("http://localhost:11434")
        url_row.addWidget(self.ollama_url, 1)
        ollama_layout.addLayout(url_row)

        # Ollama model
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.ollama_model = QLineEdit()
        self.ollama_model.setPlaceholderText("llava")
        self.ollama_model.setText("llava")
        model_row.addWidget(self.ollama_model, 1)

        # Test connection button
        self.test_btn = QPushButton("Test")
        self.test_btn.setMaximumWidth(60)
        self.test_btn.clicked.connect(self._test_ollama_connection)
        model_row.addWidget(self.test_btn)
        ollama_layout.addLayout(model_row)

        backend_layout.addWidget(self.ollama_widget)
        self.ollama_widget.hide()

        # WD14 settings (initially hidden)
        self.wd14_widget = QWidget()
        wd14_layout = QVBoxLayout(self.wd14_widget)
        wd14_layout.setContentsMargins(0, 5, 0, 0)

        # WD14 model dropdown with sizes
        wd14_model_row = QHBoxLayout()
        wd14_model_row.addWidget(QLabel("Model:"))
        self.wd14_model_combo = QComboBox()
        self.wd14_model_combo.addItem("ViT v2 (~350MB)", "vit-v2")
        self.wd14_model_combo.addItem("SwinV2 v2 (~430MB)", "swinv2-v2")
        self.wd14_model_combo.addItem("ConvNeXtV2 v2 (~680MB)", "convnextv2-v2")
        self.wd14_model_combo.addItem("MOAT v2 (~840MB)", "moat-v2")
        self.wd14_model_combo.addItem("ViT v3 (~350MB)", "vit-v3")
        self.wd14_model_combo.addItem("Custom Path...", "custom")
        self.wd14_model_combo.currentIndexChanged.connect(self._on_wd14_model_changed)
        wd14_model_row.addWidget(self.wd14_model_combo, 1)
        wd14_layout.addLayout(wd14_model_row)

        # Custom WD14 model path (hidden by default)
        self.wd14_custom_path_widget = QWidget()
        wd14_custom_path_layout = QHBoxLayout(self.wd14_custom_path_widget)
        wd14_custom_path_layout.setContentsMargins(0, 0, 0, 0)
        wd14_custom_path_layout.addWidget(QLabel("Path:"))
        self.wd14_custom_path = QLineEdit()
        self.wd14_custom_path.setPlaceholderText(
            "Path to model file (.onnx, .safetensors, .pt)"
        )
        wd14_custom_path_layout.addWidget(self.wd14_custom_path, 1)

        # Browse button
        self.wd14_browse_btn = QPushButton("Browse...")
        self.wd14_browse_btn.setMaximumWidth(80)
        self.wd14_browse_btn.clicked.connect(self._browse_wd14_model)
        wd14_custom_path_layout.addWidget(self.wd14_browse_btn)

        wd14_layout.addWidget(self.wd14_custom_path_widget)
        self.wd14_custom_path_widget.hide()

        # WD14 threshold
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold:"))
        self.wd14_threshold = QDoubleSpinBox()
        self.wd14_threshold.setRange(0.1, 0.9)
        self.wd14_threshold.setSingleStep(0.05)
        self.wd14_threshold.setValue(0.55)
        self.wd14_threshold.setToolTip(
            "Minimum confidence for tags (lower = more tags, higher = fewer)"
        )
        threshold_row.addWidget(self.wd14_threshold, 1)
        wd14_layout.addLayout(threshold_row)

        backend_layout.addWidget(self.wd14_widget)
        self.wd14_widget.hide()

        self.right_panel_layout.addWidget(backend_group)

        # Generated Caption group
        generated_group = QGroupBox("Generated Caption")
        generated_layout = QVBoxLayout(generated_group)

        # Generated caption text area
        self.generated_caption_text = QTextEdit()
        self.generated_caption_text.setPlaceholderText(
            "Generated caption will appear here. You can edit it before using the action buttons below."
        )
        self.generated_caption_text.setMinimumHeight(120)
        generated_layout.addWidget(self.generated_caption_text)

        # Generate button
        self.generate_btn = QPushButton("Generate Caption")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; }"
            "QPushButton:enabled { background-color: #2196F3; color: white; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        generated_layout.addWidget(self.generate_btn)

        # Progress label
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet(
            "QLabel { color: #666666; font-style: italic; }"
        )
        self.progress_label.hide()
        generated_layout.addWidget(self.progress_label)

        self.right_panel_layout.addWidget(generated_group)

        # Action buttons row
        actions_group = QGroupBox("Caption Actions")
        actions_layout = QVBoxLayout(actions_group)

        # Button row
        button_row = QHBoxLayout()

        self.replace_btn = QPushButton("⬇ Replace")
        self.replace_btn.setToolTip("Replace saved caption with generated caption")
        self.replace_btn.clicked.connect(self._on_replace_clicked)
        button_row.addWidget(self.replace_btn)

        self.append_btn = QPushButton("+ Append")
        self.append_btn.setToolTip("Append generated caption to saved caption")
        self.append_btn.clicked.connect(self._on_append_clicked)
        button_row.addWidget(self.append_btn)

        self.blend_btn = QPushButton("⚡ Blend")
        self.blend_btn.setToolTip("Use Ollama to blend generated and saved captions")
        self.blend_btn.clicked.connect(self._on_blend_clicked)
        button_row.addWidget(self.blend_btn)

        actions_layout.addLayout(button_row)

        # Ollama blend model selection
        blend_model_row = QHBoxLayout()
        blend_model_row.addWidget(QLabel("Blend Model:"))
        self.blend_model_combo = QComboBox()
        self.blend_model_combo.setToolTip("Ollama model to use for blending captions")
        blend_model_row.addWidget(self.blend_model_combo, 1)

        # Refresh button to update model list
        self.blend_refresh_btn = QPushButton("↻")
        self.blend_refresh_btn.setMaximumWidth(30)
        self.blend_refresh_btn.setToolTip("Refresh available Ollama models")
        self.blend_refresh_btn.clicked.connect(self._refresh_blend_models)
        blend_model_row.addWidget(self.blend_refresh_btn)
        actions_layout.addLayout(blend_model_row)

        # Ollama blend instructions
        blend_label = QLabel("Blend Instructions:")
        blend_label.setStyleSheet("QLabel { font-size: 11px; color: #666666; }")
        actions_layout.addWidget(blend_label)

        self.blend_instructions = QTextEdit()
        self.blend_instructions.setPlaceholderText(
            "Instructions for how Ollama should blend the captions..."
        )
        self.blend_instructions.setMinimumHeight(60)
        self.blend_instructions.setMaximumHeight(80)
        default_instructions = (
            "Combine these two image captions into a single, comprehensive caption. "
            "Merge complementary details and remove redundancies. "
            "Provide only the blended caption without explanation or preamble."
        )
        self.blend_instructions.setPlainText(default_instructions)
        actions_layout.addWidget(self.blend_instructions)

        # Populate blend models on startup
        QTimer.singleShot(100, self._refresh_blend_models)

        self.right_panel_layout.addWidget(actions_group)

        # Saved Caption group
        saved_group = QGroupBox("Saved Caption")
        saved_layout = QVBoxLayout(saved_group)

        # Saved caption text area
        self.saved_caption_text = QTextEdit()
        self.saved_caption_text.setPlaceholderText(
            "Saved caption from .txt file. Edit and save with Ctrl+S or the button below."
        )
        self.saved_caption_text.setMinimumHeight(120)
        self.saved_caption_text.textChanged.connect(self._on_caption_changed)
        saved_layout.addWidget(self.saved_caption_text)

        # Save button
        self.save_btn = QPushButton("Save Caption")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; }"
            "QPushButton:enabled { background-color: #4CAF50; color: white; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.save_btn.clicked.connect(self._on_save_requested)
        self.save_btn.setToolTip("Save caption (Ctrl+S)")
        saved_layout.addWidget(self.save_btn)

        self.right_panel_layout.addWidget(saved_group, 1)  # Give this stretch factor

        # Navigation bar — always clickable regardless of which widget has focus
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 4, 0, 0)
        nav_layout.setSpacing(6)

        self.nav_prev_btn = QPushButton("◀  Prev")
        self.nav_prev_btn.setToolTip("Previous image  (← when text box not focused)")
        self.nav_prev_btn.setMinimumHeight(32)
        self.nav_prev_btn.clicked.connect(self._navigate_previous)
        nav_layout.addWidget(self.nav_prev_btn)

        self.nav_label = QLabel("— / —")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setMinimumWidth(70)
        nav_layout.addWidget(self.nav_label)

        self.nav_next_btn = QPushButton("Next  ▶")
        self.nav_next_btn.setToolTip("Next image  (→ when text box not focused)")
        self.nav_next_btn.setMinimumHeight(32)
        self.nav_next_btn.clicked.connect(self._navigate_next)
        nav_layout.addWidget(self.nav_next_btn)

        self.right_panel_layout.addWidget(nav_widget)

        self.splitter.addWidget(self.right_panel)

        # Set splitter proportions (70% image, 30% panel)
        self.splitter.setSizes([700, 300])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Zoom display
        self.zoom_label = QLabel()
        self.zoom_label.setStyleSheet("QLabel { padding: 2px; }")
        self.status_bar.addPermanentWidget(self.zoom_label)

        # Check Florence2 availability
        self._check_backend_status()

    def _setup_menu(self):
        """Setup the menu bar."""
        menu_bar = self.menuBar()

        # Options menu
        options_menu = menu_bar.addMenu("&Options")

        # Fit to window
        self.fit_act = QAction("&Fit to Window", self)
        self.fit_act.setCheckable(True)
        self.fit_act.setChecked(self.fit_to_window)
        self.fit_act.triggered.connect(self.toggle_fit_mode)
        options_menu.addAction(self.fit_act)

        # Native size
        self.native_act = QAction("&Native Size", self)
        self.native_act.setCheckable(True)
        self.native_act.setChecked(not self.fit_to_window)
        self.native_act.triggered.connect(self.set_native_size)
        options_menu.addAction(self.native_act)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Save shortcut (Ctrl+S)
        self.shortcut_save = QShortcut(QKeySequence.StandardKey.Save, self)
        self.shortcut_save.activated.connect(self._on_save_requested)

    def _load_settings(self):
        """Load caption settings from parent app."""
        if not self.parent_app:
            return

        # Load backend selection
        backend = getattr(self.parent_app, "caption_backend", "florence2")
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == backend:
                self.backend_combo.setCurrentIndex(i)
                break

        # Load Florence2 settings
        florence_model = getattr(self.parent_app, "caption_florence_model", "base")
        for i in range(self.florence_model_combo.count()):
            if self.florence_model_combo.itemData(i) == florence_model:
                self.florence_model_combo.setCurrentIndex(i)
                break

        florence_detail = getattr(
            self.parent_app, "caption_florence_detail", "detailed"
        )
        for i in range(self.florence_detail_combo.count()):
            if self.florence_detail_combo.itemData(i) == florence_detail:
                self.florence_detail_combo.setCurrentIndex(i)
                break

        florence_custom_path = getattr(
            self.parent_app, "caption_florence_custom_path", ""
        )
        if florence_custom_path:
            self.florence_custom_path.setText(florence_custom_path)

        # Load Ollama settings
        ollama_url = getattr(
            self.parent_app, "caption_ollama_url", "http://localhost:11434"
        )
        self.ollama_url.setText(ollama_url)

        ollama_model = getattr(self.parent_app, "caption_ollama_model", "llava")
        self.ollama_model.setText(ollama_model)

        # Load WD14 settings
        wd14_model = getattr(self.parent_app, "caption_wd14_model", "vit-v2")
        for i in range(self.wd14_model_combo.count()):
            if self.wd14_model_combo.itemData(i) == wd14_model:
                self.wd14_model_combo.setCurrentIndex(i)
                break

        wd14_threshold = getattr(self.parent_app, "caption_wd14_threshold", 0.55)
        self.wd14_threshold.setValue(wd14_threshold)

        # Load WD14 custom paths into generator
        wd14_custom_paths = getattr(self.parent_app, "caption_wd14_custom_paths", {})
        if wd14_custom_paths:
            for model_key, path in wd14_custom_paths.items():
                self.generator.set_wd14_custom_path(model_key, path)
                # If this is the currently selected custom model, show the path in UI
                if model_key == "custom" or wd14_model == model_key:
                    self.wd14_custom_path.setText(path)

        # Load blend instructions
        blend_instructions = getattr(self.parent_app, "caption_blend_instructions", "")
        if blend_instructions:
            self.blend_instructions.setPlainText(blend_instructions)

        # Load blend model (will be set after models are loaded)
        self.caption_blend_model = getattr(self.parent_app, "caption_blend_model", "")

    def _save_settings(self):
        """Save caption settings to parent app."""
        if not self.parent_app:
            return

        # Save settings
        self.parent_app.caption_backend = self.backend_combo.currentData()
        self.parent_app.caption_ollama_url = self.ollama_url.text().strip()
        self.parent_app.caption_ollama_model = self.ollama_model.text().strip()
        self.parent_app.caption_florence_model = self.florence_model_combo.currentData()
        self.parent_app.caption_florence_detail = (
            self.florence_detail_combo.currentData()
        )
        self.parent_app.caption_florence_custom_path = (
            self.florence_custom_path.text().strip()
        )
        self.parent_app.caption_wd14_model = self.wd14_model_combo.currentData()
        self.parent_app.caption_wd14_threshold = self.wd14_threshold.value()

        # Save WD14 custom paths from generator
        self.parent_app.caption_wd14_custom_paths = dict(
            self.generator.wd14_custom_paths
        )

        # Save blend instructions
        self.parent_app.caption_blend_instructions = (
            self.blend_instructions.toPlainText().strip()
        )

        # Save blend model
        self.parent_app.caption_blend_model = self.blend_model_combo.currentText()

        # Trigger settings write
        if hasattr(self.parent_app, "write_settings"):
            self.parent_app.write_settings()

    def _navigate_previous(self):
        """Navigate to previous image."""
        if self.image_list and self._check_unsaved_changes():
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.load_image()

    def _navigate_next(self):
        """Navigate to next image."""
        if self.image_list and self._check_unsaved_changes():
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.load_image()

    def _navigate_first(self):
        """Navigate to first image."""
        if self.image_list and self._check_unsaved_changes():
            self.current_index = 0
            self.load_image()

    def _navigate_last(self):
        """Navigate to last image."""
        if self.image_list and self._check_unsaved_changes():
            self.current_index = len(self.image_list) - 1
            self.load_image()

    def _check_backend_status(self):
        """Check if local backends (Florence2, WD14) are available."""
        backend = self.backend_combo.currentData()

        if backend == "florence2":
            available, message = self.generator.is_florence_available()
            if available:
                self.backend_status.setText(
                    "Florence2 ready (model loads on first use)"
                )
                self.backend_status.setStyleSheet(
                    "QLabel { color: green; font-size: 11px; }"
                )
            else:
                self.backend_status.setText(message)
                self.backend_status.setStyleSheet(
                    "QLabel { color: red; font-size: 11px; }"
                )
                self.generate_btn.setEnabled(False)
        elif backend == "wd14":
            available, message = self.generator.is_wd14_available()
            if available:
                self.backend_status.setText("WD14 ready (model loads on first use)")
                self.backend_status.setStyleSheet(
                    "QLabel { color: green; font-size: 11px; }"
                )
            else:
                self.backend_status.setText(message)
                self.backend_status.setStyleSheet(
                    "QLabel { color: red; font-size: 11px; }"
                )
                self.generate_btn.setEnabled(False)

    def _on_backend_changed(self, index):
        """Handle backend selection change."""
        backend = self.backend_combo.currentData()
        if backend == "ollama":
            self.ollama_widget.show()
            self.wd14_widget.hide()
            self.florence_widget.hide()
            self.backend_status.hide()
            self.generate_btn.setEnabled(True)
        elif backend == "wd14":
            self.ollama_widget.hide()
            self.wd14_widget.show()
            self.florence_widget.hide()
            self.backend_status.show()
            available, _ = self.generator.is_wd14_available()
            self.generate_btn.setEnabled(available)
            self._check_backend_status()
        else:  # florence2
            self.ollama_widget.hide()
            self.wd14_widget.hide()
            self.florence_widget.show()
            self.backend_status.show()
            available, _ = self.generator.is_florence_available()
            self.generate_btn.setEnabled(available)
            self._check_backend_status()

    def _on_florence_model_changed(self, index):
        """Handle Florence model selection change."""
        model_key = self.florence_model_combo.currentData()
        if model_key == "custom":
            self.custom_path_widget.show()
        else:
            self.custom_path_widget.hide()

    def _on_wd14_model_changed(self, index):
        """Handle WD14 model selection change."""
        model_key = self.wd14_model_combo.currentData()
        if model_key == "custom":
            self.wd14_custom_path_widget.show()
        else:
            self.wd14_custom_path_widget.hide()

    def _browse_wd14_model(self):
        """Browse for WD14 model file."""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WD14 Model File",
            "",
            "Model Files (*.onnx *.safetensors *.pt *.pth *.bin);;All Files (*)",
        )
        if file_path:
            self.wd14_custom_path.setText(file_path)

    def _test_ollama_connection(self):
        """Test connection to Ollama server."""
        url = self.ollama_url.text().strip() or "http://localhost:11434"
        connected, message, models = self.generator.test_ollama_connection(url)

        if connected:
            self.status_bar.showMessage(message, 5000)
            # Auto-fill model if found
            if models and not self.ollama_model.text():
                self.ollama_model.setText(models[0])
        else:
            self.status_bar.showMessage(message, 5000)

    def _on_caption_changed(self):
        """Handle caption text change."""
        if not self.has_unsaved_changes:
            self.has_unsaved_changes = True
            self._update_window_title()

    def _on_generate_clicked(self):
        """Generate caption for current image."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        backend = self.backend_combo.currentData()

        # Disable generate button during generation
        self.generate_btn.setEnabled(False)
        self.progress_label.show()

        # Get Ollama settings
        ollama_url = self.ollama_url.text().strip() or "http://localhost:11434"
        ollama_model = self.ollama_model.text().strip() or "llava"

        # Get Florence settings
        florence_model = self.florence_model_combo.currentData()
        florence_detail = self.florence_detail_combo.currentData()

        # Set custom model path if using custom
        if florence_model == "custom":
            custom_path = self.florence_custom_path.text().strip()
            if custom_path:
                self.generator.set_custom_model_path(custom_path)

        # Get WD14 settings
        wd14_model = self.wd14_model_combo.currentData()
        wd14_threshold = self.wd14_threshold.value()

        # Set custom WD14 model path if using custom
        if wd14_model == "custom":
            custom_path = self.wd14_custom_path.text().strip()
            if custom_path:
                self.generator.set_wd14_custom_path("custom", custom_path)
            else:
                # Clear custom path if empty
                self.generator.set_wd14_custom_path("custom", "")

        # Create worker thread
        self.caption_worker = CaptionWorker(
            self.generator,
            image_path,
            backend,
            ollama_url,
            ollama_model,
            florence_model,
            florence_detail,
            wd14_model,
            wd14_threshold,
        )
        self.caption_worker.progress.connect(self._on_generation_progress)
        self.caption_worker.finished.connect(self._on_generation_finished)
        self.caption_worker.error.connect(self._on_generation_error)
        self.caption_worker.start()

    def _on_generation_progress(self, message):
        """Handle generation progress update."""
        self.progress_label.setText(message)

    def _on_generation_finished(self, caption):
        """Handle successful caption generation."""
        self.generated_caption_text.setPlainText(caption)
        self.progress_label.hide()
        self.generate_btn.setEnabled(True)
        self.status_bar.showMessage("Caption generated", 3000)

        # Properly cleanup the worker thread
        if self.caption_worker:
            self.caption_worker.wait()  # Wait for thread to fully terminate
            self.caption_worker.deleteLater()  # Schedule for safe deletion
            self.caption_worker = None

    def _on_generation_error(self, error):
        """Handle caption generation error."""
        self.progress_label.hide()
        self.generate_btn.setEnabled(True)
        self.status_bar.showMessage(f"Error: {error}", 5000)

        # Properly cleanup the worker thread
        if self.caption_worker:
            self.caption_worker.wait()  # Wait for thread to fully terminate
            self.caption_worker.deleteLater()  # Schedule for safe deletion
            self.caption_worker = None

    def _on_replace_clicked(self):
        """Replace saved caption with generated caption."""
        generated = self.generated_caption_text.toPlainText().strip()
        if not generated:
            self.status_bar.showMessage("No generated caption to replace with", 3000)
            return

        self.saved_caption_text.setPlainText(generated)
        self.has_unsaved_changes = True
        self._update_window_title()
        self.status_bar.showMessage(
            "Replaced saved caption with generated caption", 3000
        )

    def _on_append_clicked(self):
        """Append generated caption to saved caption."""
        generated = self.generated_caption_text.toPlainText().strip()
        if not generated:
            self.status_bar.showMessage("No generated caption to append", 3000)
            return

        saved = self.saved_caption_text.toPlainText().strip()
        if saved:
            # Append with newline separator if saved caption exists
            combined = saved + "\n" + generated
        else:
            # Just use generated if no saved caption
            combined = generated

        self.saved_caption_text.setPlainText(combined)
        self.has_unsaved_changes = True
        self._update_window_title()
        self.status_bar.showMessage("Appended generated caption to saved caption", 3000)

    def _detect_ollama_endpoint(self) -> tuple[bool, str, str]:
        """Detect which Ollama endpoint is available.

        Returns:
            Tuple of (success, endpoint_url, message)
        """
        import urllib.request
        import urllib.error
        import json
        import os

        # Check if debug mode is enabled
        debug = os.environ.get("IMG_SORT_DEBUG") == "1"

        # Try common Ollama endpoints in order
        endpoints = ["http://localhost:11434", "http://localhost:3000/ollama"]

        # Track first working endpoint (even if it has no models)
        fallback_endpoint = None
        fallback_message = None

        for endpoint in endpoints:
            try:
                api_url = f"{endpoint.rstrip('/')}/api/tags"
                if debug:
                    print(f"[DEBUG] Trying Ollama endpoint: {api_url}")

                request = urllib.request.Request(api_url)

                with urllib.request.urlopen(request, timeout=3) as response:
                    response_text = response.read().decode("utf-8")
                    if debug:
                        print(
                            f"[DEBUG] Response from {endpoint}: {response_text[:200]}..."
                        )

                    result = json.loads(response_text)
                    models = result.get("models", [])

                    if debug:
                        print(f"[DEBUG] Found {len(models)} models at {endpoint}")
                        if models:
                            print(
                                f"[DEBUG] First few models: {[m.get('name', 'unknown') for m in models[:3]]}"
                            )

                    if models:
                        # Found endpoint with models - use it immediately
                        if debug:
                            print(f"[DEBUG] Using endpoint {endpoint} (has models)")
                        return (
                            True,
                            endpoint,
                            f"Found {len(models)} model(s) at {endpoint}",
                        )
                    else:
                        # Endpoint works but no models - remember it as fallback
                        if fallback_endpoint is None:
                            if debug:
                                print(
                                    f"[DEBUG] Endpoint {endpoint} connected but no models, saving as fallback"
                                )
                            fallback_endpoint = endpoint
                            fallback_message = (
                                f"Connected to {endpoint} but no models installed"
                            )

            except urllib.error.HTTPError as e:
                if e.code == 401:
                    # Authentication required - this is likely Open-WebUI
                    # Try to provide helpful guidance but don't fail completely
                    if debug:
                        print(
                            f"[DEBUG] Endpoint {endpoint} requires authentication (401)"
                        )
                        print("[DEBUG] This is likely Open-WebUI with auth enabled")
                    # Don't set as fallback since it won't work without auth
                    continue
                else:
                    if debug:
                        print(
                            f"[DEBUG] HTTP Error from {endpoint}: {e.code} {e.reason}"
                        )
                    continue
            except Exception as e:
                if debug:
                    print(
                        f"[DEBUG] Failed to connect to {endpoint}: {type(e).__name__}: {e}"
                    )
                continue

        # Use fallback endpoint if we found one (even with no models)
        if fallback_endpoint:
            if debug:
                print(
                    f"[DEBUG] No endpoints with models found, using fallback: {fallback_endpoint}"
                )
            return True, fallback_endpoint, fallback_message

        # No endpoints worked at all
        if debug:
            print("[DEBUG] No Ollama endpoints responded")
        return (
            False,
            "",
            (
                "No accessible Ollama server found. Options:\n"
                "1. Start standalone Ollama: 'ollama serve' (recommended)\n"
                "2. Install Ollama from https://ollama.com\n"
                "3. If using Open-WebUI with auth, run Ollama separately"
            ),
        )

    def _refresh_blend_models(self):
        """Refresh the list of available Ollama models for blending."""
        # Save current selection (or use saved setting on first load)
        current_model = self.blend_model_combo.currentText()
        if not current_model and hasattr(self, "caption_blend_model"):
            current_model = self.caption_blend_model

        # Clear and repopulate
        self.blend_model_combo.clear()

        # Try to detect Ollama endpoint
        success, endpoint, message = self._detect_ollama_endpoint()

        if not success:
            # No Ollama server found
            self.blend_model_combo.addItem("Ollama not detected")
            self.status_bar.showMessage(message, 5000)
            return

        # Update the Ollama URL field to the detected endpoint
        if endpoint != self.ollama_url.text():
            self.ollama_url.setText(endpoint)

        try:
            import urllib.request
            import json

            # Get list of models from Ollama
            api_url = f"{endpoint.rstrip('/')}/api/tags"
            request = urllib.request.Request(api_url)

            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in result.get("models", [])]

                if models:
                    for model in sorted(models):
                        self.blend_model_combo.addItem(model)

                    # Restore previous selection if it exists
                    if current_model:
                        index = self.blend_model_combo.findText(current_model)
                        if index >= 0:
                            self.blend_model_combo.setCurrentIndex(index)

                    self.status_bar.showMessage(
                        f"Loaded {len(models)} models from {endpoint}", 3000
                    )
                else:
                    self.blend_model_combo.addItem("No models found")
                    self.status_bar.showMessage(
                        "Ollama connected but no models installed", 5000
                    )

        except Exception as e:
            self.blend_model_combo.addItem(f"Error: {str(e)[:30]}")
            self.status_bar.showMessage(f"Failed to load models: {e}", 5000)

    def _on_blend_clicked(self):
        """Blend generated and saved captions using Ollama (text-only)."""
        import os

        debug = os.environ.get("IMG_SORT_DEBUG") == "1"

        generated = self.generated_caption_text.toPlainText().strip()
        saved = self.saved_caption_text.toPlainText().strip()

        if not generated:
            self.status_bar.showMessage("No generated caption to blend", 3000)
            return

        if not saved:
            self.status_bar.showMessage(
                "No saved caption to blend with (use Replace instead)", 3000
            )
            return

        # Get blend model
        blend_model = self.blend_model_combo.currentText()
        if (
            not blend_model
            or blend_model.startswith("Error:")
            or blend_model == "No models found"
            or blend_model == "Ollama not detected"
        ):
            self.status_bar.showMessage(
                "Please select a valid Ollama model for blending", 3000
            )
            return

        # Get blend instructions
        instructions = self.blend_instructions.toPlainText().strip()
        if not instructions:
            instructions = (
                "Combine these two image captions into a single, comprehensive caption. "
                "Merge complementary details and remove redundancies. "
                "Provide only the blended caption without explanation or preamble."
            )

        # Get Ollama URL
        ollama_url = self.ollama_url.text().strip() or "http://localhost:11434"

        if debug:
            print(f"[DEBUG] Starting blend with model: {blend_model}")
            print(f"[DEBUG] Ollama URL: {ollama_url}")
            print(f"[DEBUG] Generated caption length: {len(generated)}")
            print(f"[DEBUG] Saved caption length: {len(saved)}")

        # Disable buttons during blending
        self.blend_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.replace_btn.setEnabled(False)
        self.append_btn.setEnabled(False)

        # Show progress indicator
        self.progress_label.setText("Blending captions with Ollama...")
        self.progress_label.show()
        self.status_bar.showMessage(
            f"Blending captions with {blend_model}... (this may take a moment)", 0
        )

        # Create and start blend worker thread
        self.blend_worker = BlendWorker(
            self.generator, ollama_url, blend_model, instructions, generated, saved
        )
        self.blend_worker.finished.connect(self._on_blend_finished)
        self.blend_worker.error.connect(self._on_blend_error)
        self.blend_worker.start()

    def _on_blend_finished(self, blended_text):
        """Handle successful blend completion."""
        import os

        debug = os.environ.get("IMG_SORT_DEBUG") == "1"

        if debug:
            print(f"[DEBUG] Blend completed, result length: {len(blended_text)}")

        # Save current text to undo stack before replacing
        current_text = self.generated_caption_text.toPlainText()
        if current_text:  # Only save if there was text
            self._generated_undo_stack.append(current_text)
            self._generated_redo_stack.clear()  # Clear redo on new action

        # Put blended result into the Generated Caption box (not Saved)
        self.generated_caption_text.setPlainText(blended_text)

        # Hide progress and re-enable buttons
        self.progress_label.hide()
        self.blend_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.replace_btn.setEnabled(True)
        self.append_btn.setEnabled(True)

        self.status_bar.showMessage(
            "Captions blended - result in Generated Caption box", 3000
        )

        # Cleanup worker
        if self.blend_worker:
            self.blend_worker.wait()
            self.blend_worker.deleteLater()
            self.blend_worker = None

    def _on_blend_error(self, error_msg):
        """Handle blend error."""
        import os

        debug = os.environ.get("IMG_SORT_DEBUG") == "1"

        if debug:
            print(f"[DEBUG] Blend failed: {error_msg}")

        # Hide progress and re-enable buttons
        self.progress_label.hide()
        self.blend_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.replace_btn.setEnabled(True)
        self.append_btn.setEnabled(True)

        self.status_bar.showMessage(f"Blend failed: {error_msg}", 5000)

        # Cleanup worker
        if self.blend_worker:
            self.blend_worker.wait()
            self.blend_worker.deleteLater()
            self.blend_worker = None

    def _undo_generated_caption(self):
        """Undo last change to generated caption (restore previous text)."""
        if not self._generated_undo_stack:
            self.status_bar.showMessage("Nothing to undo", 2000)
            return

        # Save current text to redo stack
        current_text = self.generated_caption_text.toPlainText()
        self._generated_redo_stack.append(current_text)

        # Restore previous text
        previous_text = self._generated_undo_stack.pop()
        self.generated_caption_text.setPlainText(previous_text)
        self.status_bar.showMessage("Undo: restored previous caption", 2000)

    def _redo_generated_caption(self):
        """Redo last undone change to generated caption."""
        if not self._generated_redo_stack:
            self.status_bar.showMessage("Nothing to redo", 2000)
            return

        # Save current text to undo stack
        current_text = self.generated_caption_text.toPlainText()
        self._generated_undo_stack.append(current_text)

        # Restore redo text
        redo_text = self._generated_redo_stack.pop()
        self.generated_caption_text.setPlainText(redo_text)
        self.status_bar.showMessage("Redo: restored blended caption", 2000)

    def _on_save_requested(self):
        """Save caption to text file."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        caption = self.saved_caption_text.toPlainText().strip()
        if not caption:
            self.status_bar.showMessage("No caption to save", 3000)
            return

        image_path = self.image_list[self.current_index]
        base_name = os.path.splitext(image_path)[0]
        caption_path = base_name + ".txt"

        try:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)

            self.has_unsaved_changes = False
            self._update_window_title()
            self.status_bar.showMessage(
                f"Saved: {os.path.basename(caption_path)}", 3000
            )

        except OSError as e:
            self.status_bar.showMessage(f"Save failed: {e}", 5000)

    def _load_existing_caption(self):
        """Load existing caption file if present."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        base_name = os.path.splitext(image_path)[0]
        caption_path = base_name + ".txt"

        self.current_caption_path = caption_path

        # Clear the generated caption box and undo/redo stacks when loading a new image
        self.generated_caption_text.clear()
        self._generated_undo_stack.clear()
        self._generated_redo_stack.clear()

        # Load saved caption
        if os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read()
                self.saved_caption_text.blockSignals(True)
                self.saved_caption_text.setPlainText(caption)
                self.saved_caption_text.blockSignals(False)
                self.has_unsaved_changes = False
            except OSError:
                self.saved_caption_text.clear()
        else:
            self.saved_caption_text.blockSignals(True)
            self.saved_caption_text.clear()
            self.saved_caption_text.blockSignals(False)
            self.has_unsaved_changes = False

        self._update_window_title()

    def _update_window_title(self):
        """Update window title and nav label with current image and unsaved indicator."""
        if self.image_list and self.current_index < len(self.image_list):
            filename = os.path.basename(self.image_list[self.current_index])
            n = len(self.image_list)
            pos = self.current_index + 1
            title = f"Auto Caption - {filename} [{pos}/{n}]"
            self.nav_label.setText(f"{pos} / {n}")
            self.nav_prev_btn.setEnabled(True)
            self.nav_next_btn.setEnabled(True)
        else:
            title = "Auto Caption Tool"
            self.nav_label.setText("— / —")
            self.nav_prev_btn.setEnabled(False)
            self.nav_next_btn.setEnabled(False)

        if self.has_unsaved_changes:
            title = "* " + title

        self.setWindowTitle(title)

    def _check_unsaved_changes(self):
        """Check for unsaved changes before navigation.

        Returns:
            True if OK to proceed, False if user cancelled
        """
        if not self.has_unsaved_changes:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved caption changes. Do you want to save them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        if reply == QMessageBox.StandardButton.Save:
            self._on_save_requested()
            return True
        elif reply == QMessageBox.StandardButton.Discard:
            self.has_unsaved_changes = False
            return True
        else:  # Cancel
            return False

    def load_image(self):
        """Load the current image."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]

        # Load pixmap
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            self.status_bar.showMessage(f"Failed to load: {image_path}", 5000)
            return

        # Update display
        self.update_image()
        self._update_window_title()

        # Load existing caption
        self._load_existing_caption()

        self.status_bar.showMessage(
            f"{os.path.basename(image_path)} - {self.original_pixmap.width()}x{self.original_pixmap.height()}",
            3000,
        )

    def update_image(self):
        """Update the displayed image based on current settings."""
        if self.original_pixmap.isNull():
            return

        if self.fit_to_window:
            # Fit to scroll area
            available_size = self.scroll_area.viewport().size()
            scaled = self.original_pixmap.scaled(
                available_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.displayed_pixmap = scaled
            # Calculate effective zoom for display
            self.zoom_factor = scaled.width() / self.original_pixmap.width()
        else:
            # Apply zoom factor - create QSize with explicit integer dimensions
            new_size = QSize(
                int(self.original_pixmap.width() * self.zoom_factor),
                int(self.original_pixmap.height() * self.zoom_factor),
            )
            scaled = self.original_pixmap.scaled(
                new_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.displayed_pixmap = scaled

        self.image_label.setPixmap(self.displayed_pixmap)
        self.update_zoom_display()

    def update_zoom_display(self):
        """Update the zoom percentage display."""
        zoom_pct = int(self.zoom_factor * 100)
        self.zoom_label.setText(f"Zoom: {zoom_pct}%")

    def toggle_fit_mode(self, checked):
        """Toggle fit-to-window mode."""
        self.fit_to_window = checked
        self.native_act.setChecked(not checked)
        self.update_image()

    def set_native_size(self):
        """Set native (100%) size mode."""
        self.fit_to_window = False
        self.fit_act.setChecked(False)
        self.native_act.setChecked(True)
        self.zoom_factor = 1.0
        self.update_image()

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        # Check if any text box has focus - if so, let it handle navigation keys normally
        text_has_focus = (
            self.generated_caption_text.hasFocus()
            or self.saved_caption_text.hasFocus()
            or self.blend_instructions.hasFocus()
        )

        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif (
            event.key() == Qt.Key.Key_S
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self._on_save_requested()
        elif event.key() == Qt.Key.Key_Left and not text_has_focus:
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = (self.current_index - 1) % len(self.image_list)
                    self.load_image()
        elif event.key() == Qt.Key.Key_Right and not text_has_focus:
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = (self.current_index + 1) % len(self.image_list)
                    self.load_image()
        elif event.key() == Qt.Key.Key_Home and not text_has_focus:
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = 0
                    self.load_image()
        elif event.key() == Qt.Key.Key_End and not text_has_focus:
            if self.image_list:
                if self._check_unsaved_changes():
                    self.current_index = len(self.image_list) - 1
                    self.load_image()
        elif (
            event.key() == Qt.Key.Key_Z
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            # Ctrl+Z - Undo for generated caption when it has focus
            if self.generated_caption_text.hasFocus():
                self._undo_generated_caption()
            else:
                super().keyPressEvent(event)
        elif (
            event.key() == Qt.Key.Key_Y
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            # Ctrl+Y - Redo for generated caption when it has focus
            if self.generated_caption_text.hasFocus():
                self._redo_generated_caption()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Switch to native mode if in fit mode
            if self.fit_to_window:
                self.fit_to_window = False
                self.zoom_factor = (
                    self.displayed_pixmap.width() / self.original_pixmap.width()
                )
                self.fit_act.setChecked(False)
                self.native_act.setChecked(True)

            # Apply zoom
            delta = event.angleDelta().y()
            self.zoom_factor *= 1.1 if delta > 0 else 0.9
            self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
            self.update_image()
            event.accept()
        else:
            super().wheelEvent(event)

    def eventFilter(self, source, event):
        """Event filter for scroll area resize handling."""
        if source == self.scroll_area and event.type() == event.Type.Resize:
            if self.fit_to_window:
                self.update_image()
        return super().eventFilter(source, event)

    def showEvent(self, event):
        """Handle window show event - initialize after window is fully visible."""
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            QTimer.singleShot(50, self._deferred_initialize)

    def _deferred_initialize(self):
        """Initialize after window geometry is finalized."""
        QApplication.processEvents()

        # Retry maximize if needed (Linux window managers can be slow)
        if not self.isMaximized() and self._maximize_attempts < 5:
            self._maximize_attempts += 1
            self.showMaximized()
            QTimer.singleShot(100, self._deferred_initialize)
            return

        self._load_saved_settings()
        self.load_image()

    def _load_saved_settings(self):
        """Load saved caption settings from the main app."""
        main_win = self.parent_app or find_main_window()
        if not main_win:
            return

        # Load backend setting
        if hasattr(main_win, "caption_backend"):
            backend = main_win.caption_backend or "florence2"
            index = self.backend_combo.findData(backend)
            if index >= 0:
                self.backend_combo.setCurrentIndex(index)

        # Load Ollama settings
        if hasattr(main_win, "caption_ollama_url") and main_win.caption_ollama_url:
            self.ollama_url.setText(main_win.caption_ollama_url)

        if hasattr(main_win, "caption_ollama_model") and main_win.caption_ollama_model:
            self.ollama_model.setText(main_win.caption_ollama_model)

        # Load Florence settings
        if hasattr(main_win, "caption_florence_model"):
            florence_model = main_win.caption_florence_model or "base"
            index = self.florence_model_combo.findData(florence_model)
            if index >= 0:
                self.florence_model_combo.setCurrentIndex(index)

        if hasattr(main_win, "caption_florence_detail"):
            florence_detail = main_win.caption_florence_detail or "detailed"
            index = self.florence_detail_combo.findData(florence_detail)
            if index >= 0:
                self.florence_detail_combo.setCurrentIndex(index)

        if (
            hasattr(main_win, "caption_florence_custom_path")
            and main_win.caption_florence_custom_path
        ):
            self.florence_custom_path.setText(main_win.caption_florence_custom_path)

    def closeEvent(self, event):
        """Handle window close - save settings and check for unsaved changes."""
        if not self._check_unsaved_changes():
            event.ignore()
            return

        # Clean up any running worker thread
        if self.caption_worker:
            self.caption_worker.wait()  # Wait for thread to fully terminate
            self.caption_worker.deleteLater()  # Schedule for safe deletion
            self.caption_worker = None

        # Save all caption settings
        self._save_settings()

        event.accept()

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        if self.fit_to_window and not self.original_pixmap.isNull():
            self.update_image()
