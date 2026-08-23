"""Tests for AdvancedFilterDialog.has_active_filters() method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogHasActiveFilters:
    """Test suite for AdvancedFilterDialog.has_active_filters() method."""

    def test_has_active_filters_no_filters(self):
        """Test has_active_filters() False when no filters are enabled."""
        dialog = AdvancedFilterDialog()
        assert dialog.has_active_filters() is False

    def test_has_active_filters_face_detection(self):
        """Test has_active_filters() True when face_detection is enabled."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"face_detection": True})
        assert dialog.has_active_filters() is True

    def test_has_active_filters_object_detection(self):
        """Test has_active_filters() True when object_detection is enabled."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"object_detection": True})
        assert dialog.has_active_filters() is True

    def test_has_active_filters_scene_recognition(self):
        """Test has_active_filters() True when scene_recognition is enabled."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"scene_recognition": True})
        assert dialog.has_active_filters() is True

    def test_has_active_filters_quality_filter(self):
        """Test has_active_filters() True when quality_filter is enabled."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"quality_filter": True})
        assert dialog.has_active_filters() is True

    def test_has_active_filters_filesize_filter(self):
        """Test has_active_filters() True when filesize_filter is enabled."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"filesize_filter": True})
        assert dialog.has_active_filters() is True

    def test_has_active_filters_multiple_filters(self):
        """Test has_active_filters() True when multiple filters are enabled."""
        dialog = AdvancedFilterDialog()
        dialog.set_filter_settings({"face_detection": True, "object_detection": True})
        assert dialog.has_active_filters() is True
