"""Tests for AdvancedFilterDialog.select_indoor_scenes method."""

from unittest.mock import patch
from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogSelectIndoorScenes:
    """Test suite for AdvancedFilterDialog.select_indoor_scenes method."""

    def test_select_indoor_scenes_checks_indoor_groups(self):
        """Test that select_indoor_scenes checks indoor scene groups."""
        dialog = AdvancedFilterDialog()
        dialog.select_indoor_scenes()

        # Assert that indoor groups are checked
        assert dialog.scene_group_cbs["indoor_home"].isChecked()
        assert dialog.scene_group_cbs["indoor_commercial"].isChecked()
        assert dialog.scene_group_cbs["entertainment"].isChecked()

    def test_select_indoor_scenes_unchecks_non_indoor_groups(self):
        """Test that select_indoor_scenes unchecks non-indoor scene groups."""
        dialog = AdvancedFilterDialog()
        dialog.select_indoor_scenes()

        # Assert that non-indoor groups are NOT checked
        assert (
            not dialog.scene_group_cbs.get("outdoor_nature", None)
            or not dialog.scene_group_cbs["outdoor_nature"].isChecked()
        )
        assert (
            not dialog.scene_group_cbs.get("outdoor_urban", None)
            or not dialog.scene_group_cbs["outdoor_urban"].isChecked()
        )
        assert (
            not dialog.scene_group_cbs.get("transportation", None)
            or not dialog.scene_group_cbs["transportation"].isChecked()
        )

    def test_select_indoor_scenes_calls_update_filter_settings(self):
        """Test that select_indoor_scenes calls update_filter_settings."""
        dialog = AdvancedFilterDialog()

        # Mock the update_filter_settings method
        with patch.object(dialog, "update_filter_settings") as mock_update:
            dialog.select_indoor_scenes()

        # Assert that update_filter_settings was called (at least once)
        mock_update.assert_called()

    def test_select_indoor_scenes_preserves_other_filter_settings(self):
        """Test that select_indoor_scenes doesn't affect other filter settings."""
        dialog = AdvancedFilterDialog()

        # Set some other filter settings in the UI
        dialog.face_enable_cb.setChecked(True)
        dialog.face_count_slider.setValue(2)
        dialog.quality_enable_cb.setChecked(True)

        # Store the initial state
        initial_face_detection = dialog.filter_settings["face_detection"]
        initial_face_count = dialog.filter_settings["face_count_min"]
        initial_quality_filter = dialog.filter_settings["quality_filter"]

        # Call select_indoor_scenes
        dialog.select_indoor_scenes()

        # Assert that other filter settings remain unchanged
        assert dialog.filter_settings["face_detection"] == initial_face_detection
        assert dialog.filter_settings["face_count_min"] == initial_face_count
        assert dialog.filter_settings["quality_filter"] == initial_quality_filter

    def test_select_indoor_scenes_initial_state(self):
        """Test that initially, no scene groups are checked."""
        dialog = AdvancedFilterDialog()

        # Assert that initially, no scene groups are checked
        for group_name, cb in dialog.scene_group_cbs.items():
            assert not cb.isChecked(), (
                f"Scene group {group_name} should not be checked initially"
            )

    def test_select_indoor_scenes_multiple_calls(self):
        """Test that select_indoor_scenes works correctly when called multiple times."""
        dialog = AdvancedFilterDialog()

        # Call select_indoor_scenes twice
        dialog.select_indoor_scenes()
        dialog.select_indoor_scenes()

        # Assert that the scene groups remain checked after the second call
        assert dialog.scene_group_cbs["indoor_home"].isChecked()
        assert dialog.scene_group_cbs["indoor_commercial"].isChecked()
        assert dialog.scene_group_cbs["entertainment"].isChecked()

    def test_select_indoor_scenes_filter_settings_updated(self):
        """Test that select_indoor_scenes updates filter_settings correctly."""
        dialog = AdvancedFilterDialog()

        # Call select_indoor_scenes
        dialog.select_indoor_scenes()

        # Get filter_settings
        filter_settings = dialog.filter_settings

        # Assert that scene_groups contains the indoor groups
        assert "indoor_home" in filter_settings["scene_groups"]
        assert "indoor_commercial" in filter_settings["scene_groups"]
        assert "entertainment" in filter_settings["scene_groups"]
