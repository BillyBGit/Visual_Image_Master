"""
Auto Caption dialog for batch captioning from thumbnail context menu.
"""

import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                                QComboBox, QLabel, QPushButton, QProgressBar,
                                QDoubleSpinBox, QLineEdit, QCheckBox,
                                QDialogButtonBox, QMessageBox, QApplication)
from PySide6.QtCore import Qt, QTimer

from caption_generator import get_caption_generator, CaptionWorker


class AutoCaptionDialog(QDialog):
    """Dialog for configuring and running batch auto-captioning."""

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.generator = get_caption_generator()
        self.caption_worker = None
        self.current_index = 0
        self.results = []  # List of (path, caption, success) tuples

        self.setWindowTitle(f"Auto Caption - {len(image_paths)} image(s)")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Backend selection
        backend_group = QGroupBox("Caption Backend")
        backend_layout = QVBoxLayout(backend_group)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Florence2 (Local)", "florence2")
        self.backend_combo.addItem("WD14 Tagger (Local)", "wd14")
        self.backend_combo.addItem("Ollama (Server)", "ollama")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        backend_row.addWidget(self.backend_combo, 1)
        backend_layout.addLayout(backend_row)

        # Florence2 settings
        self.florence_widget = QGroupBox("Florence2 Settings")
        florence_layout = QVBoxLayout(self.florence_widget)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.florence_model_combo = QComboBox()
        self.florence_model_combo.addItem("Base (~460MB)", "base")
        self.florence_model_combo.addItem("Large (~1.5GB)", "large")
        self.florence_model_combo.addItem("Base Fine-tuned (~460MB)", "base-ft")
        self.florence_model_combo.addItem("Large Fine-tuned (~1.5GB)", "large-ft")
        model_row.addWidget(self.florence_model_combo, 1)
        florence_layout.addLayout(model_row)

        detail_row = QHBoxLayout()
        detail_row.addWidget(QLabel("Detail:"))
        self.florence_detail_combo = QComboBox()
        self.florence_detail_combo.addItem("Brief (short caption)", "brief")
        self.florence_detail_combo.addItem("Standard", "standard")
        self.florence_detail_combo.addItem("Detailed (comprehensive)", "detailed")
        self.florence_detail_combo.setCurrentIndex(2)
        detail_row.addWidget(self.florence_detail_combo, 1)
        florence_layout.addLayout(detail_row)

        backend_layout.addWidget(self.florence_widget)

        # WD14 settings
        self.wd14_widget = QGroupBox("WD14 Settings")
        wd14_layout = QVBoxLayout(self.wd14_widget)

        wd14_model_row = QHBoxLayout()
        wd14_model_row.addWidget(QLabel("Model:"))
        self.wd14_model_combo = QComboBox()
        self.wd14_model_combo.addItem("ViT v2 (~350MB)", "vit-v2")
        self.wd14_model_combo.addItem("SwinV2 v2 (~430MB)", "swinv2-v2")
        self.wd14_model_combo.addItem("ConvNeXtV2 v2 (~680MB)", "convnextv2-v2")
        self.wd14_model_combo.addItem("MOAT v2 (~840MB)", "moat-v2")
        self.wd14_model_combo.addItem("ViT v3 (~350MB)", "vit-v3")
        wd14_model_row.addWidget(self.wd14_model_combo, 1)
        wd14_layout.addLayout(wd14_model_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold:"))
        self.wd14_threshold = QDoubleSpinBox()
        self.wd14_threshold.setRange(0.1, 0.9)
        self.wd14_threshold.setSingleStep(0.05)
        self.wd14_threshold.setValue(0.35)
        self.wd14_threshold.setToolTip("Minimum confidence for tags (lower = more tags)")
        threshold_row.addWidget(self.wd14_threshold, 1)
        wd14_layout.addLayout(threshold_row)

        backend_layout.addWidget(self.wd14_widget)
        self.wd14_widget.hide()

        # Ollama settings
        self.ollama_widget = QGroupBox("Ollama Settings")
        ollama_layout = QVBoxLayout(self.ollama_widget)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self.ollama_url = QLineEdit()
        self.ollama_url.setPlaceholderText("http://localhost:11434")
        self.ollama_url.setText("http://localhost:11434")
        url_row.addWidget(self.ollama_url, 1)
        ollama_layout.addLayout(url_row)

        ollama_model_row = QHBoxLayout()
        ollama_model_row.addWidget(QLabel("Model:"))
        self.ollama_model = QLineEdit()
        self.ollama_model.setPlaceholderText("llava")
        self.ollama_model.setText("llava")
        ollama_model_row.addWidget(self.ollama_model, 1)
        ollama_layout.addLayout(ollama_model_row)

        backend_layout.addWidget(self.ollama_widget)
        self.ollama_widget.hide()

        layout.addWidget(backend_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        self.overwrite_check = QCheckBox("Overwrite existing captions")
        self.overwrite_check.setChecked(False)
        options_layout.addWidget(self.overwrite_check)

        layout.addWidget(options_group)

        # Progress
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(self.progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.image_paths))
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel(f"Ready to process {len(self.image_paths)} image(s)")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(self.progress_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate Captions")
        self.generate_btn.setMinimumHeight(35)
        self.generate_btn.clicked.connect(self._start_generation)
        button_layout.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_generation)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def _on_backend_changed(self, index):
        """Handle backend selection change."""
        backend = self.backend_combo.currentData()
        self.florence_widget.setVisible(backend == "florence2")
        self.wd14_widget.setVisible(backend == "wd14")
        self.ollama_widget.setVisible(backend == "ollama")

    def _load_settings(self):
        """Load settings from main app."""
        from utils import find_main_window
        main_win = find_main_window()
        if not main_win:
            return

        # Load backend
        backend = getattr(main_win, 'caption_backend', 'florence2')
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == backend:
                self.backend_combo.setCurrentIndex(i)
                break

        # Load Florence2 settings
        florence_model = getattr(main_win, 'caption_florence_model', 'base')
        for i in range(self.florence_model_combo.count()):
            if self.florence_model_combo.itemData(i) == florence_model:
                self.florence_model_combo.setCurrentIndex(i)
                break

        florence_detail = getattr(main_win, 'caption_florence_detail', 'detailed')
        for i in range(self.florence_detail_combo.count()):
            if self.florence_detail_combo.itemData(i) == florence_detail:
                self.florence_detail_combo.setCurrentIndex(i)
                break

        # Load WD14 settings
        wd14_model = getattr(main_win, 'caption_wd14_model', 'vit-v2')
        for i in range(self.wd14_model_combo.count()):
            if self.wd14_model_combo.itemData(i) == wd14_model:
                self.wd14_model_combo.setCurrentIndex(i)
                break

        wd14_threshold = getattr(main_win, 'caption_wd14_threshold', 0.35)
        self.wd14_threshold.setValue(wd14_threshold)

        # Load Ollama settings
        ollama_url = getattr(main_win, 'caption_ollama_url', 'http://localhost:11434')
        self.ollama_url.setText(ollama_url)

        ollama_model = getattr(main_win, 'caption_ollama_model', 'llava')
        self.ollama_model.setText(ollama_model)

    def _save_settings(self):
        """Save settings to main app."""
        from utils import find_main_window
        main_win = find_main_window()
        if not main_win:
            return

        main_win.caption_backend = self.backend_combo.currentData()
        main_win.caption_florence_model = self.florence_model_combo.currentData()
        main_win.caption_florence_detail = self.florence_detail_combo.currentData()
        main_win.caption_wd14_model = self.wd14_model_combo.currentData()
        main_win.caption_wd14_threshold = self.wd14_threshold.value()
        main_win.caption_ollama_url = self.ollama_url.text().strip()
        main_win.caption_ollama_model = self.ollama_model.text().strip()

        if hasattr(main_win, 'write_settings'):
            main_win.write_settings()

    def _start_generation(self):
        """Start the caption generation process."""
        self._save_settings()

        self.current_index = 0
        self.results = []

        # Update UI
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.progress_bar.setValue(0)

        # Start processing first image
        self._process_next_image()

    def _process_next_image(self):
        """Process the next image in the queue."""
        if self.current_index >= len(self.image_paths):
            self._finish_generation()
            return

        image_path = self.image_paths[self.current_index]
        filename = os.path.basename(image_path)

        # Check if caption already exists
        base_name = os.path.splitext(image_path)[0]
        caption_path = base_name + ".txt"

        if os.path.exists(caption_path) and not self.overwrite_check.isChecked():
            # Skip existing
            self.results.append((image_path, None, "skipped"))
            self.current_index += 1
            self.progress_bar.setValue(self.current_index)
            self.progress_label.setText(f"Skipped {filename} (caption exists)")
            QTimer.singleShot(10, self._process_next_image)
            return

        self.progress_label.setText(f"Processing {filename}... ({self.current_index + 1}/{len(self.image_paths)})")

        # Get settings
        backend = self.backend_combo.currentData()
        ollama_url = self.ollama_url.text().strip() or "http://localhost:11434"
        ollama_model = self.ollama_model.text().strip() or "llava"
        florence_model = self.florence_model_combo.currentData()
        florence_detail = self.florence_detail_combo.currentData()
        wd14_model = self.wd14_model_combo.currentData()
        wd14_threshold = self.wd14_threshold.value()

        # Create worker
        self.caption_worker = CaptionWorker(
            self.generator, image_path, backend, ollama_url, ollama_model,
            florence_model, florence_detail, wd14_model, wd14_threshold
        )
        self.caption_worker.finished.connect(self._on_caption_finished)
        self.caption_worker.error.connect(self._on_caption_error)
        self.caption_worker.start()

    def _on_caption_finished(self, caption):
        """Handle successful caption generation."""
        image_path = self.image_paths[self.current_index]

        # Save caption to file
        base_name = os.path.splitext(image_path)[0]
        caption_path = base_name + ".txt"

        try:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
            self.results.append((image_path, caption, "success"))
        except OSError as e:
            self.results.append((image_path, None, f"save error: {e}"))

        # Cleanup worker
        if self.caption_worker:
            self.caption_worker.wait()
            self.caption_worker.deleteLater()
            self.caption_worker = None

        # Move to next
        self.current_index += 1
        self.progress_bar.setValue(self.current_index)
        QTimer.singleShot(10, self._process_next_image)

    def _on_caption_error(self, error):
        """Handle caption generation error."""
        image_path = self.image_paths[self.current_index]
        self.results.append((image_path, None, f"error: {error}"))

        # Cleanup worker
        if self.caption_worker:
            self.caption_worker.wait()
            self.caption_worker.deleteLater()
            self.caption_worker = None

        # Move to next
        self.current_index += 1
        self.progress_bar.setValue(self.current_index)
        QTimer.singleShot(10, self._process_next_image)

    def _cancel_generation(self):
        """Cancel the current generation."""
        if self.caption_worker:
            self.caption_worker.wait()
            self.caption_worker.deleteLater()
            self.caption_worker = None

        self._finish_generation(cancelled=True)

    def _finish_generation(self, cancelled=False):
        """Finish the generation process and show results."""
        # Re-enable UI
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.backend_combo.setEnabled(True)

        # Count results
        success_count = sum(1 for _, _, status in self.results if status == "success")
        skip_count = sum(1 for _, _, status in self.results if status == "skipped")
        error_count = sum(1 for _, _, status in self.results if status.startswith("error"))

        if cancelled:
            self.progress_label.setText(f"Cancelled. {success_count} completed, {len(self.image_paths) - self.current_index} remaining.")
        else:
            self.progress_label.setText(f"Done! {success_count} generated, {skip_count} skipped, {error_count} errors.")

        # Show summary if there were errors
        if error_count > 0:
            errors = [f"- {os.path.basename(p)}: {s}" for p, _, s in self.results if s.startswith("error")]
            error_text = "\n".join(errors[:10])
            if error_count > 10:
                error_text += f"\n... and {error_count - 10} more"

            QMessageBox.warning(
                self, "Some Errors Occurred",
                f"{error_count} image(s) failed to generate captions:\n\n{error_text}"
            )

    def closeEvent(self, event):
        """Handle dialog close."""
        if self.caption_worker:
            self.caption_worker.wait()
            self.caption_worker.deleteLater()
            self.caption_worker = None
        event.accept()
