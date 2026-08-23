"""Tests for AutoCaptionDialog.__init__ method."""

from unittest.mock import patch, MagicMock
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogInit:
    """Test suite for AutoCaptionDialog.__init__ initialization and setup."""

    def test_init_sets_image_paths(self):
        """Test that __init__ sets image_paths correctly."""
        image_paths = ["path1.jpg", "path2.jpg"]
        with patch("autocaption_dialog.get_caption_generator"):
            dialog = AutoCaptionDialog(image_paths)
            assert dialog.image_paths == image_paths

    def test_init_sets_generator(self):
        """Test that __init__ sets generator (not None)."""
        with patch("autocaption_dialog.get_caption_generator") as mock_gen:
            mock_gen.return_value = MagicMock()
            dialog = AutoCaptionDialog(["path1.jpg"])
            assert dialog.generator is not None

    def test_init_sets_initial_state(self):
        """Test that __init__ sets initial state correctly."""
        with patch("autocaption_dialog.get_caption_generator"):
            dialog = AutoCaptionDialog(["path1.jpg"])
            assert dialog.caption_worker is None
            assert dialog.current_index == 0
            assert dialog.results == []

    def test_init_sets_window_title(self):
        """Test that __init__ sets window title correctly."""
        image_paths = ["path1.jpg", "path2.jpg", "path3.jpg"]
        with patch("autocaption_dialog.get_caption_generator"):
            dialog = AutoCaptionDialog(image_paths)
            assert dialog.windowTitle() == "Auto Caption - 3 image(s)"

    def test_init_sets_minimum_width(self):
        """Test that __init__ sets minimum width correctly."""
        with patch("autocaption_dialog.get_caption_generator"):
            dialog = AutoCaptionDialog(["path1.jpg"])
            assert dialog.minimumWidth() == 450

    def test_init_sets_modal(self):
        """Test that __init__ sets window as modal."""
        with patch("autocaption_dialog.get_caption_generator"):
            dialog = AutoCaptionDialog(["path1.jpg"])
            assert dialog.isModal()

    def test_init_calls_setup_ui(self):
        """Test that __init__ calls _setup_ui method."""
        with patch("autocaption_dialog.get_caption_generator"):
            with patch.object(AutoCaptionDialog, "_setup_ui"):
                with patch.object(AutoCaptionDialog, "_load_settings"):
                    AutoCaptionDialog(["path1.jpg"])
                    AutoCaptionDialog._setup_ui.assert_called_once()

    def test_init_calls_load_settings(self):
        """Test that __init__ calls _load_settings method."""
        with patch("autocaption_dialog.get_caption_generator"):
            with patch.object(AutoCaptionDialog, "_setup_ui"):
                with patch.object(AutoCaptionDialog, "_load_settings") as mock_load:
                    AutoCaptionDialog(["path1.jpg"])
                    mock_load.assert_called_once()
