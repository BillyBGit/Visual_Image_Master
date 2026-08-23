"""Tests for AdvancedFilterDialog.select_outdoor_scenes() method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogSelectOutdoorScenes:
    """Test suite for AdvancedFilterDialog.select_outdoor_scenes() method."""

    def test_select_outdoor_scenes_checks_outdoor_groups(self):
        """Test outdoor scene groups are checked."""
        dialog = AdvancedFilterDialog()
        dialog.select_outdoor_scenes()

        # Assert that outdoor groups are checked
        assert dialog.scene_group_cbs["outdoor_nature"].isChecked()
        assert dialog.scene_group_cbs["outdoor_urban"].isChecked()

    def test_select_outdoor_scenes_unchecks_non_outdoor_groups(self):
        """Test non-outdoor scene groups are unchecked."""
        dialog = AdvancedFilterDialog()
        dialog.select_outdoor_scenes()

        # Get all non-outdoor scene groups
        outdoor_groups = {"outdoor_nature", "outdoor_urban"}
        non_outdoor_groups = [
            name for name in dialog.scene_group_cbs.keys() if name not in outdoor_groups
        ]

        # Assert that all non-outdoor groups are unchecked
        for group_name in non_outdoor_groups:
            assert not dialog.scene_group_cbs[group_name].isChecked()

    def test_select_outdoor_scenes_updates_filter_settings(self):
        """Test filter settings contain only outdoor scene groups."""
        dialog = AdvancedFilterDialog()
        dialog.select_outdoor_scenes()

        filter_settings = dialog.get_filter_settings()
        scene_groups = filter_settings.get("scene_groups", [])

        # Assert that only outdoor groups are in the filter settings
        assert set(scene_groups) == {"outdoor_nature", "outdoor_urban"}

    def test_select_outdoor_scenes_preserves_other_filter_settings(self):
        """Test other filter settings remain unchanged."""
        dialog = AdvancedFilterDialog()

        # Set some other filter settings
        dialog.face_detection = True
        dialog.quality_threshold = 0.8

        # Store original settings
        original_settings = dialog.get_filter_settings()

        # Call the method under test
        dialog.select_outdoor_scenes()

        # Get new settings
        new_settings = dialog.get_filter_settings()

        # Assert that other settings remain unchanged
        face_detect_orig = original_settings.get("face_detection")
        face_detect_new = new_settings.get("face_detection")
        face_detect_unchanged = face_detect_orig == face_detect_new

        quality_orig = original_settings.get("quality_threshold")
        quality_new = new_settings.get("quality_threshold")
        quality_unchanged = quality_orig == quality_new

        assert face_detect_unchanged
        assert quality_unchanged

    def test_select_outdoor_scenes_enables_scene_recognition(self):
        """Test scene recognition is enabled."""
        dialog = AdvancedFilterDialog()
        dialog.select_outdoor_scenes()

        # Assert that scene recognition is enabled
        assert dialog.scene_enable_cb.isChecked()

    def test_select_outdoor_scenes_works_with_empty_initial_state(self):
        """Test select_outdoor_scenes() works with empty initial state."""
        dialog = AdvancedFilterDialog()

        # Ensure no scene groups are initially selected
        for cb in dialog.scene_group_cbs.values():
            cb.setChecked(False)

        # Call the method under test
        dialog.select_outdoor_scenes()

        # Assert that only outdoor groups are selected
        assert dialog.scene_group_cbs["outdoor_nature"].isChecked()
        assert dialog.scene_group_cbs["outdoor_urban"].isChecked()

        # Assert that all other groups remain unchecked
        outdoor_groups = {"outdoor_nature", "outdoor_urban"}
        for group_name, cb in dialog.scene_group_cbs.items():
            if group_name not in outdoor_groups:
                assert not cb.isChecked()

    def test_select_outdoor_scenes_works_with_partial_initial_state(self):
        """Test select_outdoor_scenes() works with partial initial state."""
        dialog = AdvancedFilterDialog()

        # Manually check some non-outdoor scene groups
        non_outdoor_groups = [
            name
            for name in dialog.scene_group_cbs.keys()
            if name not in {"outdoor_nature", "outdoor_urban"}
        ]

        # Check first 2 non-outdoor groups
        for group_name in non_outdoor_groups[:2]:
            dialog.scene_group_cbs[group_name].setChecked(True)

        # Call the method under test
        dialog.select_outdoor_scenes()

        # Assert that only outdoor groups remain checked
        assert dialog.scene_group_cbs["outdoor_nature"].isChecked()
        assert dialog.scene_group_cbs["outdoor_urban"].isChecked()

        # Assert that all non-outdoor groups are now unchecked
        for group_name in non_outdoor_groups:
            assert not dialog.scene_group_cbs[group_name].isChecked()
