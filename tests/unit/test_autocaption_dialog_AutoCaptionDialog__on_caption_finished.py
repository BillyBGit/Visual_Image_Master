"""Tests for AutoCaptionDialog._on_caption_finished method."""

import os
import tempfile
from unittest.mock import Mock, patch
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogOnCaptionFinished:
    """Test suite for AutoCaptionDialog._on_caption_finished method."""

    def test_on_caption_finished_saves_caption(self):
        """Test that caption is saved to file with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            caption = "Test caption"
            dialog._on_caption_finished(caption)

            caption_path = os.path.join(tmpdir, "test.txt")
            assert os.path.exists(caption_path)
            with open(caption_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == caption

    def test_on_caption_finished_handles_save_error(self):
        """Test that save errors are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an unwritable directory
            unwritable_dir = os.path.join(tmpdir, "unwritable")
            os.makedirs(unwritable_dir, exist_ok=True)
            os.chmod(unwritable_dir, 0o000)

            image_path = os.path.join(unwritable_dir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            caption = "Test caption"
            dialog._on_caption_finished(caption)

            # Check that results contains a save error entry
            assert len(dialog.results) == 1
            assert dialog.results[0][2].startswith("save error:")

            # Restore permissions
            os.chmod(unwritable_dir, 0o755)

    def test_on_caption_finished_updates_results_success(self):
        """Test that results are updated with success entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            caption = "Test caption"
            dialog._on_caption_finished(caption)

            assert len(dialog.results) == 1
            assert dialog.results[0][0] == image_path
            assert dialog.results[0][1] == caption
            assert dialog.results[0][2] == "success"

    def test_on_caption_finished_updates_results_error(self):
        """Test that results are updated with error entry on save failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an unwritable directory
            unwritable_dir = os.path.join(tmpdir, "unwritable")
            os.makedirs(unwritable_dir, exist_ok=True)
            os.chmod(unwritable_dir, 0o000)

            image_path = os.path.join(unwritable_dir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            caption = "Test caption"
            dialog._on_caption_finished(caption)

            assert len(dialog.results) == 1
            assert dialog.results[0][0] == image_path
            assert dialog.results[0][1] is None
            assert dialog.results[0][2].startswith("save error:")

            # Restore permissions
            os.chmod(unwritable_dir, 0o755)

    def test_on_caption_finished_increments_index(self):
        """Test that current_index is incremented by 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            initial_index = dialog.current_index
            caption = "Test caption"
            dialog._on_caption_finished(caption)

            assert dialog.current_index == initial_index + 1

    def test_on_caption_finished_updates_progress_bar(self):
        """Test that progress_bar.setValue is called with correct value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            # Mock the progress_bar
            dialog.progress_bar = Mock()
            dialog.progress_bar.setValue = Mock()

            caption = "Test caption"
            dialog._on_caption_finished(caption)

            # Check that progress_bar.setValue was called with current_index
            assert dialog.progress_bar.setValue.called
            called_value = dialog.progress_bar.setValue.call_args[0][0]
            assert called_value == dialog.current_index

    def test_on_caption_finished_calls_process_next_image(self):
        """Test that QTimer.singleShot is called with _process_next_image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])
            dialog.caption_worker = Mock()
            dialog.caption_worker.wait = Mock()
            dialog.caption_worker.deleteLater = Mock()

            caption = "Test caption"

            with patch("autocaption_dialog.QTimer") as mock_timer:
                dialog._on_caption_finished(caption)

                # Check that QTimer.singleShot was called
                assert mock_timer.singleShot.called
                # Check that it was called with _process_next_image
                args = mock_timer.singleShot.call_args[0]
                assert args[1] == dialog._process_next_image

    def test_on_caption_finished_cleans_up_worker(self):
        """Test that caption_worker is cleaned up and set to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.jpg")
            dialog = AutoCaptionDialog([image_path])

            # Create a mock worker
            mock_worker = Mock()
            mock_worker.wait = Mock()
            mock_worker.deleteLater = Mock()
            dialog.caption_worker = mock_worker

            caption = "Test caption"
            dialog._on_caption_finished(caption)

            # Check that worker was cleaned up
            assert mock_worker.wait.called
            assert mock_worker.deleteLater.called
            assert dialog.caption_worker is None
