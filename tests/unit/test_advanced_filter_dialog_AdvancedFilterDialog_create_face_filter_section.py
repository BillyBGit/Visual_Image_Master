"""Tests for AdvancedFilterDialog.create_face_filter_section method."""

import sys
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QCheckBox,
    QRadioButton,
    QSlider,
    QLabel,
)
from advanced_filter_dialog import AdvancedFilterDialog

# Create QApplication instance for testing
app = QApplication(sys.argv)


def test_create_face_filter_section_returns_qgroupbox():
    """Test that create_face_filter_section returns a QGroupBox instance."""
    dialog = AdvancedFilterDialog()
    result = dialog.create_face_filter_section()
    assert isinstance(result, QGroupBox), (
        "create_face_filter_section should return a QGroupBox"
    )


def test_face_filter_section_has_correct_title():
    """Test that the QGroupBox has the correct title."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_face_filter_section()
    assert group.title() == "Face Detection", (
        "QGroupBox title should be 'Face Detection'"
    )


def test_face_filter_section_has_enable_checkbox():
    """Test that the section contains a QCheckBox with text 'Filter by faces'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_face_filter_section()

    # Find the checkbox in the group's layout
    checkbox = None
    for widget in group.findChildren(QCheckBox):
        if widget.text() == "Filter by faces":
            checkbox = widget
            break

    assert checkbox is not None, (
        "Should contain a QCheckBox with text 'Filter by faces'"
    )


def test_face_filter_section_has_contains_radio_button():
    """Test that the section contains a QRadioButton with text 'Contains'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_face_filter_section()

    # Find the radio button in the group's layout
    radio_button = None
    for widget in group.findChildren(QRadioButton):
        if widget.text() == "Contains":
            radio_button = widget
            break

    assert radio_button is not None, (
        "Should contain a QRadioButton with text 'Contains'"
    )
    assert radio_button.isChecked(), (
        "Contains radio button should be checked by default"
    )


def test_face_filter_section_has_not_contains_radio_button():
    """Test that the section contains a QRadioButton with text 'Does NOT contain'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_face_filter_section()

    # Find the radio button in the group's layout
    radio_button = None
    for widget in group.findChildren(QRadioButton):
        if widget.text() == "Does NOT contain":
            radio_button = widget
            break

    assert radio_button is not None, (
        "Should contain a QRadioButton with text 'Does NOT contain'"
    )


def test_face_filter_section_has_minimum_faces_slider():
    """Test that the section contains a QSlider with range 0-10 and initial value 0."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_face_filter_section()

    # Find the slider in the group's layout
    slider = None
    for widget in group.findChildren(QSlider):
        slider = widget
        break

    assert slider is not None, "Should contain a QSlider"
    assert slider.minimum() == 0, "Slider minimum should be 0"
    assert slider.maximum() == 10, "Slider maximum should be 10"
    assert slider.value() == 0, "Slider initial value should be 0"


def test_face_filter_section_has_minimum_faces_label():
    """Test that the section contains a QLabel with initial text '0'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_face_filter_section()

    # Find the label in the group's layout
    label = None
    for widget in group.findChildren(QLabel):
        if widget.text() == "0":
            label = widget
            break

    assert label is not None, "Should contain a QLabel with initial text '0'"
