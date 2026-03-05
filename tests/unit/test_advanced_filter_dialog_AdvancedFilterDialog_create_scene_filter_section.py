"""Tests for AdvancedFilterDialog.create_scene_filter_section method."""

import sys
from advanced_filter_dialog import AdvancedFilterDialog
from PySide6.QtWidgets import (
    QGroupBox,
    QCheckBox,
    QRadioButton,
    QPushButton,
    QSlider,
    QApplication,
)

# Create QApplication instance for testing
app = QApplication(sys.argv)


class TestAdvancedFilterDialogCreateSceneFilterSection:
    """Test suite for AdvancedFilterDialog.create_scene_filter_section method.""" ""

    def test_create_scene_filter_section_returns_qgroupbox(self):
        """Test that create_scene_filter_section returns a QGroupBox instance."""
        dialog = AdvancedFilterDialog()
        result = dialog.create_scene_filter_section()
        assert isinstance(result, QGroupBox), (
            "create_scene_filter_section should return a QGroupBox"
        )

    def test_create_scene_filter_section_groupbox_title(self):
        """Test that the QGroupBox has the correct title."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_scene_filter_section()
        assert group.title() == "Scene Recognition (Places365)", (
            "QGroupBox title should be 'Scene Recognition (Places365)'"
        )

    def test_create_scene_filter_section_contains_enable_checkbox(self):
        """Test that the QGroupBox contains the enable checkbox."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_scene_filter_section()

        # Find the checkbox by iterating through all widgets
        checkbox_found = False
        for widget in group.findChildren(QCheckBox):
            if widget.text() == "Filter by scene type":
                checkbox_found = True
                break

        assert checkbox_found, (
            "QGroupBox should contain a QCheckBox with text 'Filter by scene type'"
        )

    def test_create_scene_filter_section_contains_radio_buttons(self):
        """Test that the QGroupBox contains the radio buttons."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_scene_filter_section()

        contains_found = False
        not_contains_found = False

        for widget in group.findChildren(QRadioButton):
            if widget.text() == "Contains":
                contains_found = True
            elif widget.text() == "Does NOT contain":
                not_contains_found = True

        assert contains_found, (
            "QGroupBox should contain a QRadioButton with text 'Contains'"
        )
        assert not_contains_found, (
            "QGroupBox should contain a QRadioButton with text 'Does NOT contain'"
        )

    def test_create_scene_filter_section_contains_scene_group_checkboxes(self):
        """Test that the QGroupBox contains checkboxes for each scene group."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_scene_filter_section()

        # Import SCENE_GROUPS to check against it
        from advanced_filter_dialog import SCENE_GROUPS

        scene_group_checkboxes = group.findChildren(QCheckBox)
        scene_group_names = list(SCENE_GROUPS.keys())

        # Check that we have checkboxes for each scene group
        for group_name in scene_group_names:
            found = any(
                cb.text() == group_name.replace("_", " ").title()
                for cb in scene_group_checkboxes
            )
            assert found, (
                f"QGroupBox should contain a QCheckBox for scene group '{group_name}'"
            )

    def test_create_scene_filter_section_contains_quick_selection_buttons(self):
        """Test that the QGroupBox contains the quick selection buttons."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_scene_filter_section()

        indoor_found = False
        outdoor_found = False

        for widget in group.findChildren(QPushButton):
            if widget.text() == "Indoor Scenes":
                indoor_found = True
            elif widget.text() == "Outdoor Scenes":
                outdoor_found = True

        assert indoor_found, (
            "QGroupBox should contain a QPushButton with text 'Indoor Scenes'"
        )
        assert outdoor_found, (
            "QGroupBox should contain a QPushButton with text 'Outdoor Scenes'"
        )

    def test_create_scene_filter_section_contains_confidence_slider(self):
        """Test that the QGroupBox contains the confidence slider with correct settings."""
        dialog = AdvancedFilterDialog()
        group = dialog.create_scene_filter_section()

        slider = group.findChild(QSlider)
        assert slider is not None, "QGroupBox should contain a QSlider"
        assert slider.minimum() == 10, "QSlider minimum should be 10"
        assert slider.maximum() == 90, "QSlider maximum should be 90"
        assert slider.value() == 30, "QSlider default value should be 30"
