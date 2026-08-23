"""Tests for AdvancedFilterDialog.clear_all_filters() method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogClearAllFilters:
    """Test suite for AdvancedFilterDialog.clear_all_filters() method."""

    def test_clear_all_filters_resets_face_detection(self):
        """Test that clear_all_filters() resets face detection settings."""
        dialog = AdvancedFilterDialog()
        dialog.clear_all_filters()

        assert not dialog.face_enable_cb.isChecked()
        assert dialog.face_count_slider.value() == 0

    def test_clear_all_filters_resets_object_detection(self):
        """Test that clear_all_filters() resets object detection settings."""
        dialog = AdvancedFilterDialog()
        dialog.clear_all_filters()

        assert not dialog.object_enable_cb.isChecked()
        for cb in dialog.object_category_cbs.values():
            assert not cb.isChecked()
        assert dialog.object_conf_slider.value() == 50

    def test_clear_all_filters_resets_scene_recognition(self):
        """Test that clear_all_filters() resets scene recognition settings."""
        dialog = AdvancedFilterDialog()
        dialog.clear_all_filters()

        assert not dialog.scene_enable_cb.isChecked()
        for cb in dialog.scene_group_cbs.values():
            assert not cb.isChecked()
        assert dialog.scene_conf_slider.value() == 30

    def test_clear_all_filters_resets_quality_filter(self):
        """Test that clear_all_filters() resets quality filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.clear_all_filters()

        assert not dialog.quality_enable_cb.isChecked()
        assert dialog.quality_score_slider.value() == 60
        assert dialog.quality_sharp_slider.value() == 40
        assert dialog.quality_bright_min_slider.value() == 50
        assert dialog.quality_bright_max_slider.value() == 200

    def test_clear_all_filters_resets_filesize_filter(self):
        """Test that clear_all_filters() resets file size filter settings."""
        dialog = AdvancedFilterDialog()
        dialog.clear_all_filters()

        assert not dialog.filesize_enable_cb.isChecked()
        assert dialog.filesize_min_slider.value() == 0
        assert dialog.filesize_max_slider.value() == 10240

    def test_clear_all_filters_updates_filter_settings(self):
        """Test that clear_all_filters() updates filter settings to defaults.

        This test verifies that after calling clear_all_filters(),
        the get_filter_settings() method returns the expected default values.
        """
        dialog = AdvancedFilterDialog()
        dialog.clear_all_filters()

        settings = dialog.get_filter_settings()

        # Assert default values for all filter settings
        assert settings == {
            "global_invert": False,
            "face_detection": False,
            "face_count_min": 0,
            "face_inverse": False,
            "object_detection": False,
            "object_categories": [],
            "object_confidence": 0.5,
            "object_inverse": False,
            "custom_model_path": None,
            "scene_recognition": False,
            "scene_groups": [],
            "scene_confidence": 0.3,
            "scene_inverse": False,
            "quality_filter": False,
            "quality_min_score": 60,
            "quality_sharpness_min": 40,
            "quality_brightness_min": 50,
            "quality_brightness_max": 200,
            "quality_inverse": False,
            "filesize_filter": False,
            "filesize_min_kb": 0,
            "filesize_max_kb": 10240,
        }
