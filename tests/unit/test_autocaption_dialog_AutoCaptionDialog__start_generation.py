"""Tests for AutoCaptionDialog._start_generation method."""

import pytest
from unittest.mock import MagicMock, patch
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogStartGeneration:
    """Test suite for AutoCaptionDialog._start_generation method."""

    @pytest.fixture
    def mock_dialog(self):
        """Create a mock AutoCaptionDialog instance."""
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]

        # Mock the caption generator to avoid Qt initialization
        with patch("autocaption_dialog.get_caption_generator"):
            # Create the dialog without initializing Qt
            dialog = MagicMock()

            # Set up the required attributes
            dialog.image_paths = image_paths
            dialog.current_index = 5  # Start with non-zero value
            dialog.results = ["existing", "results"]  # Start with non-empty list

            # Mock UI components
            dialog.generate_btn = MagicMock()
            dialog.cancel_btn = MagicMock()
            dialog.close_btn = MagicMock()
            dialog.backend_combo = MagicMock()
            dialog.progress_bar = MagicMock()

            # Mock the methods
            dialog._save_settings = MagicMock()
            dialog._process_next_image = MagicMock()

            return dialog

    def test_start_generation_initializes_state(self, mock_dialog):
        """Test that _start_generation resets current_index and results."""
        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        assert mock_dialog.current_index == 0
        assert mock_dialog.results == []

    def test_start_generation_disables_buttons(self, mock_dialog):
        """Test that _start_generation disables/enables buttons correctly."""
        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        mock_dialog.generate_btn.setEnabled.assert_called_once_with(False)
        mock_dialog.cancel_btn.setEnabled.assert_called_once_with(True)
        mock_dialog.close_btn.setEnabled.assert_called_once_with(False)

    def test_start_generation_disables_backend_combo(self, mock_dialog):
        """Test that _start_generation disables backend_combo."""
        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        mock_dialog.backend_combo.setEnabled.assert_called_once_with(False)

    def test_start_generation_resets_progress_bar(self, mock_dialog):
        """Test that _start_generation resets progress_bar to 0."""
        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        mock_dialog.progress_bar.setValue.assert_called_once_with(0)

    def test_start_generation_calls_save_settings(self, mock_dialog):
        """Test that _start_generation calls _save_settings."""
        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        mock_dialog._save_settings.assert_called_once()

    def test_start_generation_calls_process_next_image(self, mock_dialog):
        """Test that _start_generation calls _process_next_image."""
        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        mock_dialog._process_next_image.assert_called_once()

    def test_start_generation_preserves_image_paths(self, mock_dialog):
        """Test that _start_generation doesn't modify image_paths."""
        original_paths = mock_dialog.image_paths.copy()

        # Import and call the actual method
        AutoCaptionDialog._start_generation(mock_dialog)

        assert mock_dialog.image_paths == original_paths
