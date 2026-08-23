"""Tests for CaptionWindow._navigate_next method."""

from unittest.mock import patch

from caption_window import CaptionWindow


class TestCaptionWindowNavigateNext:
    """Test suite for CaptionWindow._navigate_next method."""

    def test_navigate_next_increments_index(self):
        """Test that _navigate_next increments current_index."""
        window = CaptionWindow(["img1.jpg", "img2.jpg", "img3.jpg"], 0)
        window._navigate_next()
        assert window.current_index == 1

    def test_navigate_next_wraps_around(self):
        """Test that _navigate_next wraps around to 0 when at the end."""
        window = CaptionWindow(["img1.jpg", "img2.jpg", "img3.jpg"], 2)
        window._navigate_next()
        assert window.current_index == 0

    def test_navigate_next_does_nothing_with_empty_list(self):
        """Test that _navigate_next does nothing with empty image_list."""
        window = CaptionWindow([], 0)
        original_index = window.current_index
        window._navigate_next()
        assert window.current_index == original_index

    def test_navigate_next_calls_load_image(self):
        """Test that _navigate_next calls load_image."""
        window = CaptionWindow(["img1.jpg"], 0)
        with patch.object(window, "load_image") as mock_load:
            window._navigate_next()
            mock_load.assert_called_once()

    def test_navigate_next_checks_unsaved_changes(self):
        """Test that _navigate_next checks for unsaved changes."""
        window = CaptionWindow(["img1.jpg"], 0)
        with patch.object(window, "_check_unsaved_changes", return_value=False):
            with patch.object(window, "load_image") as mock_load:
                window._navigate_next()
                assert window.current_index == 0
                mock_load.assert_not_called()

    def test_navigate_next_with_single_image(self):
        """Test that _navigate_next wraps around with single image."""
        window = CaptionWindow(["img1.jpg"], 0)
        window._navigate_next()
        assert window.current_index == 0

    def test_navigate_next_with_none_image_list(self):
        """Test that _navigate_next handles None image_list."""
        window = CaptionWindow(None, 0)
        window._navigate_next()  # Should not raise exception
        assert window.current_index == 0
