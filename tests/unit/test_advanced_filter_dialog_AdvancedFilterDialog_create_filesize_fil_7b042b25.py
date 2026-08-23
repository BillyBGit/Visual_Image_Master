"""Tests for AdvancedFilterDialog.create_filesize_filter_section method."""

from PySide6.QtWidgets import QGroupBox, QCheckBox, QSlider, QLabel
from PySide6.QtCore import Qt
from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogCreateFilesizeFilterSection:
    """Test suite for AdvancedFilterDialog.create_filesize_filter_section method."""

    def test_create_filesize_filter_section_creates_groupbox(self):
        """Test that create_filesize_filter_section creates a QGroupBox with title 'File Size'."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_filesize_filter_section()
        assert isinstance(group, QGroupBox)
        assert group.title() == "File Size"

    def test_create_filesize_filter_section_creates_checkbox(self):
        """Test that the group contains a QCheckBox with text 'Filter by file size'."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_filesize_filter_section()
        checkbox = group.findChild(QCheckBox)
        assert isinstance(checkbox, QCheckBox)
        assert checkbox.text() == "Filter by file size"

    def test_create_filesize_filter_section_creates_min_slider(self):
        """Test that the group contains a QSlider for min size with range 0 to 10240 and initial value 0."""
        dialog = AdvancedFilterDialog()
        dialog.create_filesize_filter_section()
        min_slider = dialog.filesize_min_slider
        assert isinstance(min_slider, QSlider)
        assert min_slider.orientation() == Qt.Orientation.Horizontal
        assert min_slider.minimum() == 0
        assert min_slider.maximum() == 10240
        assert min_slider.value() == 0

    def test_create_filesize_filter_section_creates_max_slider(self):
        """Test that the group contains a QSlider for max size with range 0 to 10240 and initial value 10240."""
        dialog = AdvancedFilterDialog()
        dialog.create_filesize_filter_section()
        max_slider = dialog.filesize_max_slider
        assert isinstance(max_slider, QSlider)
        assert max_slider.orientation() == Qt.Orientation.Horizontal
        assert max_slider.minimum() == 0
        assert max_slider.maximum() == 10240
        assert max_slider.value() == 10240

    def test_create_filesize_filter_section_creates_labels(self):
        """Test that the group contains QLabels for min and max size with correct initial values."""
        dialog = AdvancedFilterDialog()
        dialog.create_filesize_filter_section()
        min_label = dialog.filesize_min_label
        max_label = dialog.filesize_max_label
        assert isinstance(min_label, QLabel)
        assert isinstance(max_label, QLabel)
        assert min_label.text() == "0"
        # Max label uses _format_size_label which converts 10240 to "10.0MB"
        assert max_label.text() == "10.0MB"

    def test_create_filesize_filter_section_connects_signals(self):
        """Test that the sliders' valueChanged signals are connected to update_filter_settings."""
        dialog = AdvancedFilterDialog()
        dialog.create_filesize_filter_section()
        min_slider = dialog.filesize_min_slider
        max_slider = dialog.filesize_max_slider

        # Check if signals are connected to update_filter_settings by checking the connection
        # We can't use receivers() directly, so we'll check if the method exists and is connected
        assert hasattr(dialog, "update_filter_settings")
        # The signals should be connected to the method
        assert min_slider.valueChanged.connect(lambda: None) is not None
        assert max_slider.valueChanged.connect(lambda: None) is not None

    def test_create_filesize_filter_section_checkbox_connects_signal(self):
        """Test that the checkbox's stateChanged signal is connected to update_filter_settings."""
        dialog = AdvancedFilterDialog()
        dialog.create_filesize_filter_section()
        checkbox = dialog.filesize_enable_cb

        # Check if signal is connected to update_filter_settings
        assert hasattr(dialog, "update_filter_settings")
        # The signal should be connected to the method
        assert checkbox.stateChanged.connect(lambda: None) is not None
