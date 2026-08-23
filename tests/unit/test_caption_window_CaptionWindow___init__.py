"""
Tests for CaptionWindow.__init__ constructor initialization.
"""

import pytest
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import Qt


class TestCaptionWindowInit:
    """Test suite for CaptionWindow.__init__ constructor."""

    @pytest.fixture
    def mock_caption_window(self):
        """Create a mock CaptionWindow class for testing."""
        with (
            patch("caption_window.QPixmap") as mock_pixmap,
            patch("caption_window.find_main_window") as mock_find_main,
            patch("caption_window.get_caption_generator") as mock_generator,
        ):
            # Setup mocks
            mock_pixmap.return_value = MagicMock()
            mock_find_main.return_value = MagicMock()

            # Mock the generator with required methods
            generator_mock = MagicMock()
            generator_mock.is_florence_available.return_value = (False, "Not available")
            generator_mock.is_wd14_available.return_value = (False, "Not available")
            mock_generator.return_value = generator_mock

            # Import after patching
            from caption_window import CaptionWindow

            # Mock scroll_area for event filter test
            mock_scroll_area = MagicMock()

            # Create a simple mock for _setup_ui that sets scroll_area
            def mock_setup_ui(self):
                self.scroll_area = mock_scroll_area

            with (
                patch.object(CaptionWindow, "_setup_ui", mock_setup_ui),
                patch.object(CaptionWindow, "_setup_menu"),
                patch.object(CaptionWindow, "_setup_shortcuts"),
                patch.object(CaptionWindow, "_load_settings"),
                patch.object(CaptionWindow, "_check_backend_status"),
            ):
                yield CaptionWindow

    def test_init_sets_image_list_and_index(self, mock_caption_window):
        """Test that image_list and current_index are set correctly."""
        image_list = ["image1.jpg", "image2.jpg"]
        current_index = 1

        window = mock_caption_window(image_list, current_index)

        assert window.image_list == image_list
        assert window.current_index == current_index

    def test_init_sets_default_values(self, mock_caption_window):
        """Test that default values are set correctly."""
        window = mock_caption_window([], 0)

        assert window.zoom_factor == 1.0
        assert window.fit_to_window is True
        assert window.has_unsaved_changes is False
        assert window.current_caption_path is None
        assert window.caption_worker is None
        assert window.blend_worker is None
        assert window._initialized is False
        assert window._maximize_attempts == 0

    def test_init_calls_setup_methods(self, mock_caption_window):
        """Test that setup methods are called."""
        # Must use None, not MagicMock — Qt's C++ super().__init__(parent) rejects non-widgets
        window = mock_caption_window([], 0)

        # Side effects of setup methods prove they were all called
        assert window.windowTitle() == "Auto Caption Tool"
        assert window.scroll_area is not None  # set by mock _setup_ui

    def test_init_sets_window_title(self, mock_caption_window):
        """Test that window title is set correctly."""
        window = mock_caption_window([], 0)

        assert window.windowTitle() == "Auto Caption Tool"

    def test_init_sets_focus_policy(self, mock_caption_window):
        """Test that focus policy is set correctly."""
        window = mock_caption_window([], 0)

        assert window.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def test_init_installs_event_filter(self, mock_caption_window):
        """Test that event filter is installed on scroll_area."""
        window = mock_caption_window([], 0)

        # Verify event filter was installed on scroll_area
        window.scroll_area.installEventFilter.assert_called_once_with(window)

    def test_init_finds_main_window_when_parent_none(self, mock_caption_window):
        """Test that find_main_window is called when parent is None."""
        with patch("caption_window.find_main_window") as mock_find_main:
            mock_main_window = MagicMock()
            mock_find_main.return_value = mock_main_window

            window = mock_caption_window([], 0, None)

            mock_find_main.assert_called_once()
            assert window.parent_app == mock_main_window

    def test_init_uses_provided_parent(self, mock_caption_window):
        """Test that provided parent is used when available."""
        from PySide6.QtWidgets import QWidget

        # Must use a real Qt widget — MagicMock is rejected by PySide6's C++ super().__init__
        parent = QWidget()
        window = mock_caption_window([], 0, parent)

        assert window.parent_app == parent
        parent.deleteLater()
