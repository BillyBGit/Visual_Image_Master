"""Tests for AdvancedFilterDialog.create_object_filter_section method."""

import sys
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QCheckBox,
    QRadioButton,
    QSlider,
    QLineEdit,
    QPushButton,
)
from advanced_filter_dialog import AdvancedFilterDialog

# Ensure QApplication exists for testing
app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()


def test_create_object_filter_section_returns_qgroupbox():
    """Test that create_object_filter_section returns a QGroupBox instance."""
    dialog = AdvancedFilterDialog()
    result = dialog.create_object_filter_section()
    assert isinstance(result, QGroupBox)


def test_create_object_filter_section_has_enable_checkbox():
    """Test that the group has a QCheckBox with text 'Filter by objects'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    # Find the checkbox by iterating through all widgets
    checkbox = None
    for widget in group.findChildren(QCheckBox):
        if widget.text() == "Filter by objects":
            checkbox = widget
            break

    assert checkbox is not None, "Could not find 'Filter by objects' checkbox"


def test_create_object_filter_section_has_contains_radio():
    """Test that the group has a QRadioButton with text 'Contains'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    # Find the radio button by iterating through all widgets
    radio = None
    for widget in group.findChildren(QRadioButton):
        if widget.text() == "Contains":
            radio = widget
            break

    assert radio is not None, "Could not find 'Contains' radio button"


def test_create_object_filter_section_has_not_contains_radio():
    """Test that the group has a QRadioButton with text 'Does NOT contain'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    # Find the radio button by iterating through all widgets
    radio = None
    for widget in group.findChildren(QRadioButton):
        if widget.text() == "Does NOT contain":
            radio = widget
            break

    assert radio is not None, "Could not find 'Does NOT contain' radio button"


def test_create_object_filter_section_has_category_checkboxes():
    """Test that the group has QCheckBox for each category in OBJECT_CATEGORIES."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    # Import OBJECT_CATEGORIES to check against
    from advanced_filter_dialog import OBJECT_CATEGORIES

    # Find all checkboxes in the group
    checkboxes = group.findChildren(QCheckBox)

    # Check that each category has a corresponding checkbox
    for category in OBJECT_CATEGORIES.keys():
        category_name = category.replace("_", " ").title()
        found = False
        for cb in checkboxes:
            if cb.text() == category_name:
                found = True
                break
        assert found, f"Could not find checkbox for category '{category}'"


def test_create_object_filter_section_has_confidence_slider():
    """Test that the group has QSlider with range 10-90 and default value 50."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    slider = group.findChild(QSlider)
    assert slider is not None, "Could not find QSlider"
    assert slider.minimum() == 10, "Slider minimum should be 10"
    assert slider.maximum() == 90, "Slider maximum should be 90"
    assert slider.value() == 50, "Slider default value should be 50"


def test_create_object_filter_section_has_custom_model_input():
    """Test that the group has QLineEdit for custom model path."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    line_edit = group.findChild(QLineEdit)
    assert line_edit is not None, "Could not find QLineEdit for custom model path"


def test_create_object_filter_section_has_browse_button():
    """Test that the group has QPushButton with text 'Browse...'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    # Find the browse button
    browse_btn = None
    for widget in group.findChildren(QPushButton):
        if widget.text() == "Browse...":
            browse_btn = widget
            break

    assert browse_btn is not None, "Could not find 'Browse...' button"


def test_create_object_filter_section_has_clear_button():
    """Test that the group has QPushButton with text 'Clear'."""
    dialog = AdvancedFilterDialog()
    group = dialog.create_object_filter_section()

    # Find the clear button
    clear_btn = None
    for widget in group.findChildren(QPushButton):
        if widget.text() == "Clear":
            clear_btn = widget
            break

    assert clear_btn is not None, "Could not find 'Clear' button"
