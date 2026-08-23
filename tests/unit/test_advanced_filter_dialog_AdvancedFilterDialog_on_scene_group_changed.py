"""Tests for AdvancedFilterDialog.on_scene_group_changed method."""

from unittest.mock import patch
from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogOnSceneGroupChanged:
    """Test suite for AdvancedFilterDialog.on_scene_group_changed method."""

    def test_on_scene_group_changed_auto_enables_scene_detection(self):
        """Test auto-enable scene detection when group is checked."""
        dialog = AdvancedFilterDialog()

        # Check one scene group checkbox
        for cb in dialog.scene_group_cbs.values():
            cb.setChecked(True)
            break

        # Call the method
        dialog.on_scene_group_changed()

        # Assert scene_enable_cb is checked
        assert dialog.scene_enable_cb.isChecked() is True

    def test_on_scene_group_changed_does_not_disable_scene_detection(self):
        """Test that scene detection remains enabled if already enabled."""
        dialog = AdvancedFilterDialog()

        # Set scene_enable_cb to True initially
        dialog.scene_enable_cb.setChecked(True)

        # Check one scene group checkbox
        for cb in dialog.scene_group_cbs.values():
            cb.setChecked(True)
            break

        # Call the method
        dialog.on_scene_group_changed()

        # Assert scene_enable_cb remains True
        assert dialog.scene_enable_cb.isChecked() is True

    def test_on_scene_group_changed_updates_filter_settings(self):
        """Test that update_filter_settings is called."""
        dialog = AdvancedFilterDialog()

        # Mock the update_filter_settings method
        with patch.object(dialog, "update_filter_settings") as mock_update:
            dialog.on_scene_group_changed()
            mock_update.assert_called_once()

    def test_on_scene_group_changed_with_no_checked_groups(self):
        """Test scene detection not enabled when no groups are checked."""
        dialog = AdvancedFilterDialog()

        # Ensure no scene group checkboxes are checked
        for cb in dialog.scene_group_cbs.values():
            cb.setChecked(False)

        # Call the method
        dialog.on_scene_group_changed()

        # Assert scene_enable_cb is not checked
        assert dialog.scene_enable_cb.isChecked() is False

    def test_on_scene_group_changed_with_multiple_groups(self):
        """Test scene detection enabled when multiple groups are checked."""
        dialog = AdvancedFilterDialog()

        # Check multiple scene group checkboxes
        checked_count = 0
        for cb in dialog.scene_group_cbs.values():
            if checked_count < 2:  # Check first 2
                cb.setChecked(True)
                checked_count += 1
            else:
                cb.setChecked(False)

        # Call the method
        dialog.on_scene_group_changed()

        # Assert scene_enable_cb is checked
        assert dialog.scene_enable_cb.isChecked() is True

    def test_on_scene_group_changed_with_custom_model_path(self):
        """Test that custom model path is included in filter settings."""
        dialog = AdvancedFilterDialog()

        # Set custom model path
        test_path = "/path/to/custom/model"
        dialog.custom_model_path_edit.setText(test_path)

        # Check one scene group checkbox
        for cb in dialog.scene_group_cbs.values():
            cb.setChecked(True)
            break

        # Call the method
        dialog.on_scene_group_changed()

        # Assert filter_settings contains the custom_model_path
        assert hasattr(dialog, "filter_settings")
        assert "custom_model_path" in dialog.filter_settings
        assert dialog.filter_settings["custom_model_path"] == test_path

    def test_on_scene_group_changed_with_confidence_threshold(self):
        """Test that confidence threshold is included in filter settings."""
        dialog = AdvancedFilterDialog()

        # Set confidence threshold
        test_confidence = 75
        dialog.scene_conf_slider.setValue(test_confidence)

        # Check one scene group checkbox
        for cb in dialog.scene_group_cbs.values():
            cb.setChecked(True)
            break

        # Call the method
        dialog.on_scene_group_changed()

        # Assert filter_settings contains the correct confidence value
        assert hasattr(dialog, "filter_settings")
        assert "scene_confidence" in dialog.filter_settings
        # The value is stored as a percentage (0-1), so 75 becomes 0.75
        expected_confidence = test_confidence / 100
        assert dialog.filter_settings["scene_confidence"] == expected_confidence
