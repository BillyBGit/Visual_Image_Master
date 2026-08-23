"""Tests for AutoCaptionDialog._on_caption_error method."""

from unittest.mock import Mock, patch
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogOnCaptionError:
    """Test suite for AutoCaptionDialog._on_caption_error method."""

    def test_on_caption_error_records_error(self):
        """Test that error is recorded in self.results with correct image path and error message."""
        # Setup
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Call the method
        test_error = "Failed to generate caption"
        dialog._on_caption_error(test_error)

        # Assert
        assert len(dialog.results) == 1
        assert dialog.results[0][0] == "/path/to/image1.jpg"
        assert dialog.results[0][1] is None
        assert dialog.results[0][2] == "error: {}".format(test_error)

    def test_on_caption_error_increments_index(self):
        """Test that self.current_index is incremented by 1 after calling _on_caption_error."""
        # Setup
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Call the method
        dialog._on_caption_error("Test error")

        # Assert
        assert dialog.current_index == 1

    def test_on_caption_error_updates_progress_bar(self):
        """Test that self.progress_bar.setValue is called with the updated current_index."""
        # Setup
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Call the method
        dialog._on_caption_error("Test error")

        # Assert
        dialog.progress_bar.setValue.assert_called_once_with(1)

    def test_on_caption_error_cleans_up_worker(self):
        """Test that if self.caption_worker exists, it is cleaned up properly."""
        # Setup
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Create a mock worker
        worker_mock = Mock()
        dialog.caption_worker = worker_mock

        # Call the method
        dialog._on_caption_error("Test error")

        # Assert
        worker_mock.wait.assert_called_once()
        worker_mock.deleteLater.assert_called_once()
        assert dialog.caption_worker is None

    def test_on_caption_error_schedules_next_image(self):
        """Test that QTimer.singleShot is called with _process_next_image."""
        # Setup
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Mock QTimer
        with patch("autocaption_dialog.QTimer") as mock_timer:
            # Call the method
            dialog._on_caption_error("Test error")

            # Assert
            mock_timer.singleShot.assert_called_once_with(
                10, dialog._process_next_image
            )

    def test_on_caption_error_with_no_worker(self):
        """Test the function when self.caption_worker is None."""
        # Setup
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []
        dialog.caption_worker = None

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Call the method - should not raise any errors
        dialog._on_caption_error("Test error")

        # Assert
        assert dialog.current_index == 1
        assert len(dialog.results) == 1

    def test_on_caption_error_with_multiple_errors(self):
        """Test multiple errors and verify state is updated correctly."""
        # Setup
        image_paths = [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg",
            "/path/to/image3.jpg",
        ]
        dialog = AutoCaptionDialog(image_paths)
        dialog.current_index = 0
        dialog.results = []

        # Mock the progress bar
        dialog.progress_bar = Mock()

        # Call the method multiple times
        dialog._on_caption_error("Error 1")
        dialog._on_caption_error("Error 2")
        dialog._on_caption_error("Error 3")

        # Assert
        assert len(dialog.results) == 3
        assert dialog.results[0][0] == "/path/to/image1.jpg"
        assert dialog.results[0][2] == "error: Error 1"
        assert dialog.results[1][0] == "/path/to/image2.jpg"
        assert dialog.results[1][2] == "error: Error 2"
        assert dialog.results[2][0] == "/path/to/image3.jpg"
        assert dialog.results[2][2] == "error: Error 3"
        assert dialog.current_index == 3
