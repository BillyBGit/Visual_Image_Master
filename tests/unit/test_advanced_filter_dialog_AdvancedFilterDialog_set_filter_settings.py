"""Tests for AdvancedFilterDialog.set_filter_settings() method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogSetFilterSettings:
    """Test suite for AdvancedFilterDialog.set_filter_settings method."""

    def test_set_filter_settings_empty_dict(self):
        """Test with empty dict - should not change default settings."""
        dialog = AdvancedFilterDialog()
        # Store initial state
        initial_global_invert = dialog.global_invert_cb.isChecked()
        initial_face_enabled = dialog.face_enable_cb.isChecked()
        initial_object_enabled = dialog.object_enable_cb.isChecked()

        # Call with empty dict
        dialog.set_filter_settings({})

        # Assert no changes
        assert dialog.global_invert_cb.isChecked() == initial_global_invert
        assert dialog.face_enable_cb.isChecked() == initial_face_enabled
        assert dialog.object_enable_cb.isChecked() == initial_object_enabled

    def test_set_filter_settings_global_invert(self):
        """Test global_invert setting."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"global_invert": True})
        assert dialog.global_invert_cb.isChecked() is True

    def test_set_filter_settings_face_detection(self):
        """Test face detection settings."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings(
            {"face_detection": True, "face_count_min": 2, "face_inverse": True}
        )
        assert dialog.face_enable_cb.isChecked() is True
        assert dialog.face_count_slider.value() == 2
        assert dialog.face_not_contains_rb.isChecked() is True

    def test_set_filter_settings_object_detection(self):
        """Test object detection settings."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings(
            {
                "object_detection": True,
                "object_categories": ["people", "vehicles"],
                "object_confidence": 0.7,
                "object_inverse": False,
                "custom_model_path": "/path/to/model.pt",
            }
        )
        assert dialog.object_enable_cb.isChecked() is True
        assert dialog.object_category_cbs["people"].isChecked() is True
        assert dialog.object_category_cbs["vehicles"].isChecked() is True
        assert dialog.object_conf_slider.value() == 70
        assert dialog.object_contains_rb.isChecked() is True
        assert dialog.custom_model_path_edit.text() == "/path/to/model.pt"

    def test_set_filter_settings_scene_recognition(self):
        """Test scene recognition settings."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings(
            {
                "scene_recognition": True,
                "scene_groups": ["indoor_home", "outdoor_nature"],
                "scene_confidence": 0.4,
                "scene_inverse": True,
            }
        )
        assert dialog.scene_enable_cb.isChecked() is True
        assert dialog.scene_group_cbs["indoor_home"].isChecked() is True
        assert dialog.scene_group_cbs["outdoor_nature"].isChecked() is True
        assert dialog.scene_conf_slider.value() == 40
        assert dialog.scene_not_contains_rb.isChecked() is True

    def test_set_filter_settings_quality_filter(self):
        """Test quality filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings(
            {
                "quality_filter": True,
                "quality_min_score": 70,
                "quality_sharpness_min": 50,
                "quality_brightness_min": 60,
                "quality_brightness_max": 180,
                "quality_inverse": False,
            }
        )
        assert dialog.quality_enable_cb.isChecked() is True
        assert dialog.quality_score_slider.value() == 70
        assert dialog.quality_sharp_slider.value() == 50
        assert dialog.quality_bright_min_slider.value() == 60
        assert dialog.quality_bright_max_slider.value() == 180
        assert dialog.quality_meets_rb.isChecked() is True

    def test_set_filter_settings_filesize_filter(self):
        """Test file size filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings(
            {"filesize_filter": True, "filesize_min_kb": 1024, "filesize_max_kb": 5120}
        )
        assert dialog.filesize_enable_cb.isChecked() is True
        assert dialog.filesize_min_slider.value() == 1024
        assert dialog.filesize_max_slider.value() == 5120

    def test_set_filter_settings_partial_settings(self):
        """Test partial settings - only enabled filters should be set."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"face_detection": True, "object_detection": False})
        assert dialog.face_enable_cb.isChecked() is True
        assert dialog.object_enable_cb.isChecked() is False
