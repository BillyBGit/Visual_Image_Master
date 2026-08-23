"""Tests for AutoCaptionDialog._finish_generation method."""

from unittest.mock import patch
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogFinishGeneration:
    """Test suite for AutoCaptionDialog._finish_generation method."""

    def test_finish_generation_with_successful_results(self):
        """Test _finish_generation with successful caption generation."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["image1.jpg", "image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set results with successful captions
        dialog.results = [
            ("image1.jpg", "caption1", "success"),
            ("image2.jpg", "caption2", "success"),
        ]
        dialog.current_index = 2

        # Call _finish_generation
        dialog._finish_generation()

        # Assert progress_label text
        assert "Done! 2 generated, 0 skipped, 0 errors." in dialog.progress_label.text()

    def test_finish_generation_with_skipped_results(self):
        """Test _finish_generation with skipped captions."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["image1.jpg", "image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set results with skipped captions
        dialog.results = [("image1.jpg", "", "skipped"), ("image2.jpg", "", "skipped")]
        dialog.current_index = 2

        # Call _finish_generation
        dialog._finish_generation()

        # Assert progress_label text
        assert "Done! 0 generated, 2 skipped, 0 errors." in dialog.progress_label.text()

    def test_finish_generation_with_error_results(self):
        """Test _finish_generation with error status."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["image1.jpg", "image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set results with error status
        dialog.results = [
            ("image1.jpg", "", "error: failed to generate"),
            ("image2.jpg", "", "error: failed to generate"),
        ]
        dialog.current_index = 2

        # Mock QMessageBox.warning to avoid blocking
        with patch("autocaption_dialog.QMessageBox.warning") as mock_warning:
            mock_warning.return_value = None  # Avoid blocking
            # Call _finish_generation
            dialog._finish_generation()

            # Assert progress_label text
            assert (
                "Done! 0 generated, 0 skipped, 2 errors."
                in dialog.progress_label.text()
            )

    def test_finish_generation_with_cancelled_status(self):
        """Test _finish_generation with cancelled status."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["image1.jpg", "image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set results with partial completion
        dialog.results = [("image1.jpg", "caption1", "success")]
        dialog.current_index = 1

        # Call _finish_generation with cancelled=True
        dialog._finish_generation(cancelled=True)

        # Assert progress_label text
        assert "Cancelled. 1 completed, 1 remaining." in dialog.progress_label.text()

    def test_finish_generation_re_enables_ui(self):
        """Test _finish_generation re-enables UI elements."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["image1.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Disable buttons before calling _finish_generation
        dialog.generate_btn.setEnabled(False)
        dialog.cancel_btn.setEnabled(True)
        dialog.close_btn.setEnabled(False)
        dialog.backend_combo.setEnabled(False)

        # Call _finish_generation
        dialog._finish_generation()

        # Assert buttons are re-enabled
        assert dialog.generate_btn.isEnabled()
        assert not dialog.cancel_btn.isEnabled()
        assert dialog.close_btn.isEnabled()
        assert dialog.backend_combo.isEnabled()

    def test_finish_generation_shows_error_message_box(self):
        """Test _finish_generation shows error message box."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = ["image1.jpg", "image2.jpg"]
        dialog = AutoCaptionDialog(image_paths)

        # Set results with error status
        dialog.results = [
            ("image1.jpg", "", "error: failed to generate"),
            ("image2.jpg", "", "error: failed to generate"),
        ]
        dialog.current_index = 2

        # Mock QMessageBox.warning to avoid blocking
        with patch("autocaption_dialog.QMessageBox.warning") as mock_warning:
            mock_warning.return_value = None  # Avoid blocking
            # Call _finish_generation
            dialog._finish_generation()

            # Assert QMessageBox.warning was called
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args
            assert "2 image(s) failed to generate captions:" in call_args[0][2]

    def test_finish_generation_truncates_long_error_list(self):
        """Test _finish_generation truncates long error list."""
        # Create AutoCaptionDialog with mock image paths
        image_paths = [f"image{i}.jpg" for i in range(15)]
        dialog = AutoCaptionDialog(image_paths)

        # Set results with many error statuses
        dialog.results = [
            (f"image{i}.jpg", "", f"error: failed to generate {i}") for i in range(15)
        ]
        dialog.current_index = 15

        # Mock QMessageBox.warning to avoid blocking
        with patch("autocaption_dialog.QMessageBox.warning") as mock_warning:
            mock_warning.return_value = None  # Avoid blocking
            # Call _finish_generation
            dialog._finish_generation()

            # Assert error text is truncated
            call_args = mock_warning.call_args
            error_text = call_args[0][2]
            assert "... and 5 more" in error_text
