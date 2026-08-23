"""Tests for AutoCaptionDialog._process_next_image method."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication
from autocaption_dialog import AutoCaptionDialog


@pytest.fixture(scope="session", autouse=True)
def qt_app():
    """Qt application fixture for testing."""
    app = QApplication.instance() or QApplication([])
    yield app


class TestAutoCaptionDialogProcessNextImage:
    """Test suite for AutoCaptionDialog._process_next_image method."""

    def test_process_next_image_finishes_when_no_images(self):
        """Test that _finish_generation is called when no images are available."""
        dialog = AutoCaptionDialog([])

        # Mock the _finish_generation method
        dialog._finish_generation = MagicMock()

        # Call the method
        dialog._process_next_image()

        # Assert that _finish_generation was called
        dialog._finish_generation.assert_called_once()

    def test_process_next_image_skips_existing_caption(self):
        """Test that existing captions are skipped when overwrite_check is not checked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image and caption files
            image_path = os.path.join(tmpdir, "test.jpg")
            caption_path = os.path.join(tmpdir, "test.txt")

            # Create empty files
            with open(image_path, "w") as f:
                f.write("")
            with open(caption_path, "w") as f:
                f.write("")

            # Create dialog with the image path
            dialog = AutoCaptionDialog([image_path])

            # Call the method
            dialog._process_next_image()

            # Assert that the result contains the skipped entry
            assert (image_path, None, "skipped") in dialog.results

    def test_process_next_image_processes_new_image(self):
        """Test that a new image is processed when no caption exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image file
            image_path = os.path.join(tmpdir, "test.jpg")
            with open(image_path, "w") as f:
                f.write("")

            # Create dialog with the image path
            dialog = AutoCaptionDialog([image_path])

            # Mock the CaptionWorker
            with patch("autocaption_dialog.CaptionWorker") as mock_worker:
                # Call the method
                dialog._process_next_image()

                # Assert that CaptionWorker was created and started
                mock_worker.assert_called_once()
                worker_instance = mock_worker.return_value
                worker_instance.start.assert_called_once()

    def test_process_next_image_updates_progress(self):
        """Test that progress bar and label are updated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image file
            image_path = os.path.join(tmpdir, "test.jpg")
            with open(image_path, "w") as f:
                f.write("")

            # Create dialog with the image path
            dialog = AutoCaptionDialog([image_path])

            # Mock the CaptionWorker to prevent actual processing
            with patch("autocaption_dialog.CaptionWorker"):
                # Call the method
                dialog._process_next_image()

                # Assert that progress label text is updated (progress bar value is only updated after processing)
                expected_text = "Processing test.jpg... (1/1)"
                assert dialog.progress_label.text() == expected_text

    def test_process_next_image_handles_overwrite_check(self):
        """Test that overwrite_check determines whether to skip existing captions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image and caption files
            image_path = os.path.join(tmpdir, "test.jpg")
            caption_path = os.path.join(tmpdir, "test.txt")

            # Create empty files
            with open(image_path, "w") as f:
                f.write("")
            with open(caption_path, "w") as f:
                f.write("")

            # Create dialog with the image path
            dialog = AutoCaptionDialog([image_path])

            # Set overwrite_check to True
            dialog.overwrite_check.setChecked(True)

            # Mock the CaptionWorker
            with patch("autocaption_dialog.CaptionWorker") as mock_worker:
                # Call the method
                dialog._process_next_image()

                # Assert that CaptionWorker was created and started (overwrite is enabled)
                mock_worker.assert_called_once()
                worker_instance = mock_worker.return_value
                worker_instance.start.assert_called_once()

    def test_process_next_image_collects_worker_settings(self):
        """Test that worker is created with correct settings from UI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image file
            image_path = os.path.join(tmpdir, "test.jpg")
            with open(image_path, "w") as f:
                f.write("")

            # Create dialog with the image path
            dialog = AutoCaptionDialog([image_path])

            # Set some UI values using the correct API
            # Find index for backend
            backend_index = None
            for i in range(dialog.backend_combo.count()):
                if dialog.backend_combo.itemData(i) == "florence2":
                    backend_index = i
                    break
            if backend_index is not None:
                dialog.backend_combo.setCurrentIndex(backend_index)

            dialog.ollama_url.setText("http://test:1234")
            dialog.ollama_model.setText("test_model")

            # Find index for florence model
            florence_model_index = None
            for i in range(dialog.florence_model_combo.count()):
                if dialog.florence_model_combo.itemData(i) == "base":
                    florence_model_index = i
                    break
            if florence_model_index is not None:
                dialog.florence_model_combo.setCurrentIndex(florence_model_index)

            # Find index for florence detail
            florence_detail_index = None
            for i in range(dialog.florence_detail_combo.count()):
                if dialog.florence_detail_combo.itemData(i) == "brief":
                    florence_detail_index = i
                    break
            if florence_detail_index is not None:
                dialog.florence_detail_combo.setCurrentIndex(florence_detail_index)

            # Find index for wd14 model
            wd14_model_index = None
            for i in range(dialog.wd14_model_combo.count()):
                if dialog.wd14_model_combo.itemData(i) == "vit-v2":
                    wd14_model_index = i
                    break
            if wd14_model_index is not None:
                dialog.wd14_model_combo.setCurrentIndex(wd14_model_index)

            dialog.wd14_threshold.setValue(0.5)

            # Mock the CaptionWorker
            with patch("autocaption_dialog.CaptionWorker") as mock_worker:
                # Call the method
                dialog._process_next_image()

                # Assert that CaptionWorker was created with correct parameters
                mock_worker.assert_called_once()
                call_args = mock_worker.call_args

                # Check the arguments passed to CaptionWorker
                # The arguments are: generator, image_path, backend, ollama_url, ollama_model, florence_model, florence_detail, wd14_model, wd14_threshold
                positional_args = call_args[0]
                assert len(positional_args) == 9
                assert (
                    positional_args[1] == image_path
                )  # image_path is the 2nd argument
                assert positional_args[2] == "florence2"  # backend is the 3rd argument
                assert (
                    positional_args[3] == "http://test:1234"
                )  # ollama_url is the 4th argument
                assert (
                    positional_args[4] == "test_model"
                )  # ollama_model is the 5th argument
                assert (
                    positional_args[5] == "base"
                )  # florence_model is the 6th argument
                assert (
                    positional_args[6] == "brief"
                )  # florence_detail is the 7th argument
                assert positional_args[7] == "vit-v2"  # wd14_model is the 8th argument
                assert positional_args[8] == 0.5  # wd14_threshold is the 9th argument

    def test_process_next_image_connects_worker_signals(self):
        """Test that worker signals are connected to appropriate slots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image file
            image_path = os.path.join(tmpdir, "test.jpg")
            with open(image_path, "w") as f:
                f.write("")

            # Create dialog with the image path
            dialog = AutoCaptionDialog([image_path])

            # Mock the CaptionWorker
            with patch("autocaption_dialog.CaptionWorker") as mock_worker:
                # Call the method
                dialog._process_next_image()

                # Get the worker instance
                worker_instance = mock_worker.return_value

                # Assert that finished signal is connected to _on_caption_finished
                worker_instance.finished.connect.assert_any_call(
                    dialog._on_caption_finished
                )

                # Assert that error signal is connected to _on_caption_error
                worker_instance.error.connect.assert_any_call(dialog._on_caption_error)
