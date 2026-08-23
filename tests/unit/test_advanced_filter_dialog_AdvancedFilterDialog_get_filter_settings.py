"""Tests for AdvancedFilterDialog.get_filter_settings() method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogGetFilterSettings:
    """Test suite for AdvancedFilterDialog.get_filter_settings method."""

    def test_get_filter_settings_returns_copy(self):
        """Test that get_filter_settings returns a copy, not the original dict."""
        dialog = AdvancedFilterDialog()
        settings = dialog.get_filter_settings()

        assert isinstance(settings, dict)
        assert settings is not dialog.filter_settings

    def test_get_filter_settings_default_values(self):
        """Test that get_filter_settings returns default values."""
        dialog = AdvancedFilterDialog()
        settings = dialog.get_filter_settings()

        assert settings == dialog.filter_settings

    def test_get_filter_settings_after_update(self):
        """Test that get_filter_settings reflects changes after update_filter_settings."""
        dialog = AdvancedFilterDialog()

        # Update some settings by changing UI state
        dialog.face_enable_cb.setChecked(True)
        dialog.object_enable_cb.setChecked(False)
        dialog.update_filter_settings()
        settings = dialog.get_filter_settings()

        assert settings["face_detection"] is True
        assert settings["object_detection"] is False

    def test_get_filter_settings_face_detection(self):
        """Test get_filter_settings with face_detection enabled."""
        dialog = AdvancedFilterDialog()
        dialog.face_enable_cb.setChecked(True)
        dialog.update_filter_settings()
        settings = dialog.get_filter_settings()

        assert settings["face_detection"] is True

    def test_get_filter_settings_object_detection(self):
        """Test get_filter_settings with object_detection enabled."""
        dialog = AdvancedFilterDialog()
        dialog.object_enable_cb.setChecked(True)
        dialog.update_filter_settings()
        settings = dialog.get_filter_settings()

        assert settings["object_detection"] is True

    def test_get_filter_settings_scene_recognition(self):
        """Test get_filter_settings with scene_recognition enabled."""
        dialog = AdvancedFilterDialog()
        dialog.scene_enable_cb.setChecked(True)
        dialog.update_filter_settings()
        settings = dialog.get_filter_settings()

        assert settings["scene_recognition"] is True

    def test_get_filter_settings_quality_filter(self):
        """Test get_filter_settings with quality_filter enabled."""
        dialog = AdvancedFilterDialog()
        dialog.quality_enable_cb.setChecked(True)
        dialog.update_filter_settings()
        settings = dialog.get_filter_settings()

        assert settings["quality_filter"] is True, (
            "quality_filter should be True when enabled"
        )

    def test_get_filter_settings_filesize_filter(self):
        """Test get_filter_settings with filesize_filter enabled."""
        dialog = AdvancedFilterDialog()
        dialog.filesize_enable_cb.setChecked(True)
        dialog.update_filter_settings()
        settings = dialog.get_filter_settings()

        assert settings["filesize_filter"] is True, (
            "filesize_filter should be True when enabled"
        )
