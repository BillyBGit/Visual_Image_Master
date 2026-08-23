"""Tests for AutoCaptionDialog._cancel_generation method."""

from unittest.mock import Mock
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogCancelGeneration:
    """Test suite for AutoCaptionDialog._cancel_generation method."""

    def test_cancel_generation_with_active_worker(self):
        """Test cancel generation with active worker."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Create a real mock object and store reference
        worker_mock = Mock()
        dialog.caption_worker = worker_mock

        # Mock _finish_generation to avoid actual UI updates
        dialog._finish_generation = Mock()

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert caption_worker.wait() and caption_worker.deleteLater() were called
        worker_mock.wait.assert_called_once()
        worker_mock.deleteLater.assert_called_once()

        # Assert caption_worker is set to None
        assert dialog.caption_worker is None

        # Assert _finish_generation is called with cancelled=True
        dialog._finish_generation.assert_called_once_with(cancelled=True)

    def test_cancel_generation_without_worker(self):
        """Test cancel generation without worker."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set caption_worker to None
        dialog.caption_worker = None

        # Mock _finish_generation to avoid actual UI updates
        dialog._finish_generation = Mock()

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert no errors occur
        assert True

        # Assert _finish_generation is called with cancelled=True
        dialog._finish_generation.assert_called_once_with(cancelled=True)

    def test_cancel_generation_updates_ui_state(self):
        """Test cancel generation updates UI state."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Create a real mock object and store reference
        worker_mock = Mock()
        dialog.caption_worker = worker_mock

        # Mock UI buttons
        dialog.generate_btn = Mock()
        dialog.cancel_btn = Mock()
        dialog.close_btn = Mock()
        dialog.backend_combo = Mock()
        dialog.progress_label = Mock()

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert UI buttons are re-enabled
        dialog.generate_btn.setEnabled.assert_called_once_with(True)
        dialog.cancel_btn.setEnabled.assert_called_once_with(False)
        dialog.close_btn.setEnabled.assert_called_once_with(True)
        dialog.backend_combo.setEnabled.assert_called_once_with(True)

    def test_cancel_generation_calls_finish_generation(self):
        """Test cancel generation calls _finish_generation."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set caption_worker to a mock object
        dialog.caption_worker = Mock()

        # Mock _finish_generation to avoid actual UI updates
        dialog._finish_generation = Mock()

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert _finish_generation is called with cancelled=True
        dialog._finish_generation.assert_called_once_with(cancelled=True)

    def test_cancel_generation_with_partial_results(self):
        """Test cancel generation with partial results."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set caption_worker to a mock object
        dialog.caption_worker = Mock()

        # Mock _finish_generation to avoid actual UI updates
        dialog._finish_generation = Mock()

        # Populate results with some entries (path, caption, status format)
        dialog.results = [
            ("/path/to/image1.jpg", "Caption 1", "success"),
            ("/path/to/image2.jpg", None, "processing"),  # Partial result
        ]

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert results are preserved
        assert dialog.results == [
            ("/path/to/image1.jpg", "Caption 1", "success"),
            ("/path/to/image2.jpg", None, "processing"),
        ]

        # Assert _finish_generation is called with cancelled=True
        dialog._finish_generation.assert_called_once_with(cancelled=True)

    def test_cancel_generation_cleanup_worker(self):
        """Test cancel generation properly cleans up worker."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Create a real mock object and store reference
        worker_mock = Mock()
        dialog.caption_worker = worker_mock

        # Mock _finish_generation to avoid actual UI updates
        dialog._finish_generation = Mock()

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert caption_worker is properly cleaned up
        worker_mock.wait.assert_called_once()
        worker_mock.deleteLater.assert_called_once()
        assert dialog.caption_worker is None

        # Assert _finish_generation is called with cancelled=True
        dialog._finish_generation.assert_called_once_with(cancelled=True)

    def test_cancel_generation_interrupts_processing(self):
        """Test cancel generation interrupts processing."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set caption_worker to a mock object
        dialog.caption_worker = Mock()

        # Mock _finish_generation to avoid actual UI updates
        dialog._finish_generation = Mock()

        # Set current_index to a value > 0
        dialog.current_index = 1

        # Call _cancel_generation
        dialog._cancel_generation()

        # Assert processing stops
        assert dialog.current_index == 1  # Should remain unchanged

        # Assert _finish_generation is called with cancelled=True
        dialog._finish_generation.assert_called_once_with(cancelled=True)
