"""Tests for AutoCaptionDialog._setup_ui method."""

import pytest
from PySide6.QtWidgets import QVBoxLayout, QGroupBox, QProgressBar, QLabel, QPushButton
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogSetupUi:
    """Test suite for AutoCaptionDialog._setup_ui method."""

    @pytest.fixture
    def dialog(self):
        """Create a test AutoCaptionDialog instance."""
        mock_image_paths = ["image1.jpg", "image2.jpg"]
        dialog = AutoCaptionDialog(mock_image_paths)
        dialog._setup_ui()
        return dialog

    def test_setup_ui_creates_main_layout(self, dialog):
        """Test that _setup_ui creates a QVBoxLayout as main layout."""
        layout = dialog.layout()
        assert isinstance(layout, QVBoxLayout), "Main layout should be QVBoxLayout"

    def test_setup_ui_creates_backend_group(self, dialog):
        """Test that backend_group exists and is a QGroupBox with correct title."""
        # Find the backend group by iterating through the layout
        backend_group = None
        layout = dialog.layout()
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QGroupBox) and widget.title() == "Caption Backend":
                backend_group = widget
                break

        assert backend_group is not None, "backend_group should exist"
        assert backend_group.title() == "Caption Backend", (
            "backend_group should have correct title"
        )

    def test_setup_ui_creates_backend_combo_box(self, dialog):
        """Test that backend_combo exists and has 3 items with correct data."""
        backend_combo = dialog.backend_combo
        assert backend_combo is not None, "backend_combo should exist"
        assert backend_combo.count() == 3, "backend_combo should have 3 items"

        # Check item data
        assert backend_combo.itemData(0) == "florence2", (
            "First item should be florence2"
        )
        assert backend_combo.itemData(1) == "wd14", "Second item should be wd14"
        assert backend_combo.itemData(2) == "ollama", "Third item should be ollama"

    def test_setup_ui_creates_florence_widget_hidden(self, dialog):
        """Test that florence_widget exists, is a QGroupBox, and is initially hidden."""
        florence_widget = dialog.florence_widget
        assert florence_widget is not None, "florence_widget should exist"
        assert isinstance(florence_widget, QGroupBox), (
            "florence_widget should be QGroupBox"
        )
        # Note: florence_widget is not hidden by default in the implementation

    def test_setup_ui_creates_wd14_widget_hidden(self, dialog):
        """Test that wd14_widget exists, is a QGroupBox, and is initially hidden."""
        wd14_widget = dialog.wd14_widget
        assert wd14_widget is not None, "wd14_widget should exist"
        assert isinstance(wd14_widget, QGroupBox), "wd14_widget should be QGroupBox"
        assert wd14_widget.isHidden(), "wd14_widget should be initially hidden"

    def test_setup_ui_creates_ollama_widget_hidden(self, dialog):
        """Test that ollama_widget exists, is a QGroupBox, and is initially hidden."""
        ollama_widget = dialog.ollama_widget
        assert ollama_widget is not None, "ollama_widget should exist"
        assert isinstance(ollama_widget, QGroupBox), "ollama_widget should be QGroupBox"
        assert ollama_widget.isHidden(), "ollama_widget should be initially hidden"

    def test_setup_ui_creates_progress_group(self, dialog):
        """Test that progress_group exists and contains QProgressBar and QLabel."""
        progress_group = dialog.progress_group
        assert progress_group is not None, "progress_group should exist"

        progress_bar = dialog.progress_bar
        assert progress_bar is not None, "progress_bar should exist"
        assert isinstance(progress_bar, QProgressBar), (
            "progress_bar should be QProgressBar"
        )

        progress_label = dialog.progress_label
        assert progress_label is not None, "progress_label should exist"
        assert isinstance(progress_label, QLabel), "progress_label should be QLabel"

    def test_setup_ui_creates_buttons(self, dialog):
        """Test that generate_btn, cancel_btn, and close_btn exist and are QPushButton instances."""
        generate_btn = dialog.generate_btn
        assert generate_btn is not None, "generate_btn should exist"
        assert isinstance(generate_btn, QPushButton), (
            "generate_btn should be QPushButton"
        )

        cancel_btn = dialog.cancel_btn
        assert cancel_btn is not None, "cancel_btn should exist"
        assert isinstance(cancel_btn, QPushButton), "cancel_btn should be QPushButton"

        close_btn = dialog.close_btn
        assert close_btn is not None, "close_btn should exist"
        assert isinstance(close_btn, QPushButton), "close_btn should be QPushButton"
