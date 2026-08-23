"""Tests for AdvancedFilterDialog.update_filter_settings() method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogUpdateFilterSettings:
    """Test suite for AdvancedFilterDialog.update_filter_settings method."""

    def test_update_filter_settings_global_invert(self):
        """Test global invert filter setting."""
        dialog = AdvancedFilterDialog()
        dialog.global_invert_cb.setChecked(True)
        dialog.update_filter_settings()
        assert dialog.filter_settings["global_invert"] is True

    def test_update_filter_settings_face_detection(self):
        """Test face detection filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.face_enable_cb.setChecked(True)
        dialog.face_count_slider.setValue(3)
        dialog.face_not_contains_rb.setChecked(True)
        dialog.update_filter_settings()
        assert dialog.filter_settings["face_detection"] is True
        assert dialog.filter_settings["face_count_min"] == 3
        assert dialog.filter_settings["face_inverse"] is True

    def test_update_filter_settings_object_detection(self):
        """Test object detection filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.object_enable_cb.setChecked(True)
        dialog.object_conf_slider.setValue(75)
        dialog.object_not_contains_rb.setChecked(True)
        dialog.custom_model_path_edit.setText("/path/to/model.pt")
        dialog.update_filter_settings()
        assert dialog.filter_settings["object_detection"] is True
        assert dialog.filter_settings["object_confidence"] == 0.75
        assert dialog.filter_settings["object_inverse"] is True
        assert dialog.filter_settings["custom_model_path"] == "/path/to/model.pt"

    def test_update_filter_settings_scene_recognition(self):
        """Test scene recognition filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.scene_enable_cb.setChecked(True)
        dialog.scene_conf_slider.setValue(40)
        dialog.scene_not_contains_rb.setChecked(True)
        dialog.update_filter_settings()
        assert dialog.filter_settings["scene_recognition"] is True
        assert dialog.filter_settings["scene_confidence"] == 0.4
        assert dialog.filter_settings["scene_inverse"] is True

    def test_update_filter_settings_quality_filter(self):
        """Test quality filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.quality_enable_cb.setChecked(True)
        dialog.quality_score_slider.setValue(80)
        dialog.quality_sharp_slider.setValue(60)
        dialog.quality_bright_min_slider.setValue(70)
        dialog.quality_bright_max_slider.setValue(180)
        dialog.quality_not_meets_rb.setChecked(True)
        dialog.update_filter_settings()
        assert dialog.filter_settings["quality_filter"] is True
        assert dialog.filter_settings["quality_min_score"] == 80
        assert dialog.filter_settings["quality_sharpness_min"] == 60
        assert dialog.filter_settings["quality_brightness_min"] == 70
        assert dialog.filter_settings["quality_brightness_max"] == 180
        assert dialog.filter_settings["quality_inverse"] is True

    def test_update_filter_settings_filesize_filter(self):
        """Test file size filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.filesize_enable_cb.setChecked(True)
        dialog.filesize_min_slider.setValue(1024)
        dialog.filesize_max_slider.setValue(5120)
        dialog.update_filter_settings()
        assert dialog.filter_settings["filesize_filter"] is True
        assert dialog.filter_settings["filesize_min_kb"] == 1024
        assert dialog.filter_settings["filesize_max_kb"] == 5120

    def test_update_filter_settings_object_categories(self):
        """Test object categories selection."""
        dialog = AdvancedFilterDialog()
        dialog.object_enable_cb.setChecked(True)

        # Create mock checkbox objects
        class MockCheckbox:
            def __init__(self, checked=False):
                self._checked = checked

            def isChecked(self):
                return self._checked

            def setChecked(self, checked):
                self._checked = checked

        # Mock the object_category_cbs dictionary
        dialog.object_category_cbs = {
            "person": MockCheckbox(),
            "car": MockCheckbox(),
            "dog": MockCheckbox(),
        }
        # Set checked state for person and car
        dialog.object_category_cbs["person"].setChecked(True)
        dialog.object_category_cbs["car"].setChecked(True)
        dialog.object_category_cbs["dog"].setChecked(False)
        dialog.update_filter_settings()
        assert dialog.filter_settings["object_categories"] == ["person", "car"]

    def test_update_filter_settings_scene_groups(self):
        """Test scene groups selection."""
        dialog = AdvancedFilterDialog()
        dialog.scene_enable_cb.setChecked(True)

        # Create mock checkbox objects
        class MockCheckbox:
            def __init__(self, checked=False):
                self._checked = checked

            def isChecked(self):
                return self._checked

            def setChecked(self, checked):
                self._checked = checked

        # Mock the scene_group_cbs dictionary
        dialog.scene_group_cbs = {
            "indoor_home": MockCheckbox(),
            "outdoor_nature": MockCheckbox(),
            "city_street": MockCheckbox(),
        }
        # Set checked state for indoor_home and outdoor_nature
        dialog.scene_group_cbs["indoor_home"].setChecked(True)
        dialog.scene_group_cbs["outdoor_nature"].setChecked(True)
        dialog.scene_group_cbs["city_street"].setChecked(False)
        dialog.update_filter_settings()
        assert dialog.filter_settings["scene_groups"] == [
            "indoor_home",
            "outdoor_nature",
        ]
