"""Tests for AdvancedFilterDialog._format_size_label method."""

from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogFormatSizeLabel:
    """Test suite for AdvancedFilterDialog._format_size_label method."""

    def test_format_size_label_less_than_1024_kb(self):
        """Test formatting when size is less than 1024 KB."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(512)
        assert result == "512KB"

    def test_format_size_label_exactly_1024_kb(self):
        """Test formatting when size is exactly 1024 KB."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(1024)
        assert result == "1.0MB"

    def test_format_size_label_greater_than_1024_kb(self):
        """Test formatting when size is greater than 1024 KB."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(2048)
        assert result == "2.0MB"

    def test_format_size_label_with_decimal_kb(self):
        """Test formatting when size has decimal KB value."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(1536)
        assert result == "1.5MB"

    def test_format_size_label_zero_kb(self):
        """Test formatting when size is 0 KB."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(0)
        assert result == "0KB"

    def test_format_size_label_large_value(self):
        """Test formatting with a large KB value."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(5120)
        assert result == "5.0MB"

    def test_format_size_label_negative_value(self):
        """Test formatting with a negative KB value (no validation)."""
        dialog = AdvancedFilterDialog()
        result = dialog._format_size_label(-100)
        assert result == "-100KB"
