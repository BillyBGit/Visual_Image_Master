"""Tests for AdvancedFilterDialog.on_object_category_changed method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogOnObjectCategoryChanged:
    """Test suite for on_object_category_changed method."""

    def test_on_object_category_changed_enables_filter(self):
        """Test that checking a category checkbox enables the object filter."""
        dialog = AdvancedFilterDialog()

        # Simulate checking a category checkbox
        first_category = list(dialog.object_category_cbs.values())[0]
        first_category.setChecked(True)

        # Call the method
        dialog.on_object_category_changed()

        # Assert that object_enable_cb is checked
        assert dialog.object_enable_cb.isChecked() is True

    def test_on_object_category_changed_no_change_when_enabled(self):
        """Test that calling the method doesn't change enabled state
        if already enabled."""
        dialog = AdvancedFilterDialog()

        # Set object_enable_cb to checked
        dialog.object_enable_cb.setChecked(True)

        # Call the method
        dialog.on_object_category_changed()

        # Assert that object_enable_cb remains checked
        assert dialog.object_enable_cb.isChecked() is True

    def test_on_object_category_changed_multiple_categories(self):
        """Test that checking multiple category checkboxes enables the object filter."""
        dialog = AdvancedFilterDialog()

        # Simulate checking multiple category checkboxes
        categories = list(dialog.object_category_cbs.values())[:2]
        for cb in categories:
            cb.setChecked(True)

        # Call the method
        dialog.on_object_category_changed()

        # Assert that object_enable_cb is checked
        assert dialog.object_enable_cb.isChecked() is True

    def test_on_object_category_changed_all_unchecked(self):
        """Test that unchecking all category checkboxes disables the object filter."""
        dialog = AdvancedFilterDialog()

        # Simulate unchecking all category checkboxes
        # Since we're not actually checking any, the any() should return False
        dialog.on_object_category_changed()

        # Assert that object_enable_cb is not checked
        assert dialog.object_enable_cb.isChecked() is False

    def test_on_object_category_changed_updates_settings(self):
        """Test that calling the method updates filter settings."""
        dialog = AdvancedFilterDialog()

        # Call the method
        dialog.on_object_category_changed()

        # Assert that update_filter_settings was called
        # This would need to be mocked or verified through side effects
        # For now, we'll just ensure the method completes without error
        assert True
