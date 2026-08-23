"""Tests for CaptionWorker.run method."""

from unittest.mock import MagicMock
from caption_generator import CaptionWorker, CaptionGenerator


class TestCaptionWorkerRun:
    """Test suite for CaptionWorker.run method."""

    def test_run_with_florence2_backend(self):
        """Test CaptionWorker.run with florence2 backend."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (True, "OK")
        generator.generate_florence.return_value = "test caption"

        worker = CaptionWorker(
            generator=generator, image_path="/fake/path.jpg", backend="florence2"
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify finished signal was emitted with correct caption
        worker.finished.emit.assert_called_once_with("test caption")
        worker.error.emit.assert_not_called()

    def test_run_with_wd14_backend(self):
        """Test CaptionWorker.run with wd14 backend."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (True, "OK")
        generator.generate_wd14.return_value = "test caption"

        worker = CaptionWorker(
            generator=generator, image_path="/fake/path.jpg", backend="wd14"
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify finished signal was emitted with correct caption
        worker.finished.emit.assert_called_once_with("test caption")
        worker.error.emit.assert_not_called()

    def test_run_with_ollama_backend(self):
        """Test CaptionWorker.run with ollama backend."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (True, "OK")
        generator.generate_ollama.return_value = "test caption"

        worker = CaptionWorker(
            generator=generator,
            image_path="/fake/path.jpg",
            backend="ollama",
            ollama_url="http://localhost:11434",
            ollama_model="llama2",
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify finished signal was emitted with correct caption
        worker.finished.emit.assert_called_once_with("test caption")
        worker.error.emit.assert_not_called()

    def test_run_with_invalid_image(self):
        """Test CaptionWorker.run with invalid image."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (False, "Invalid image")

        worker = CaptionWorker(
            generator=generator, image_path="/fake/path.jpg", backend="florence2"
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify error signal was emitted with correct message
        worker.error.emit.assert_called_once_with(
            "Image validation failed: Invalid image"
        )
        worker.finished.emit.assert_not_called()

    def test_run_with_exception(self):
        """Test CaptionWorker.run with exception during generation."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (True, "OK")
        generator.generate_florence.side_effect = Exception("Test error")

        worker = CaptionWorker(
            generator=generator, image_path="/fake/path.jpg", backend="florence2"
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify error signal was emitted with exception message
        worker.error.emit.assert_called_once_with("Test error")

    def test_run_progress_signals(self):
        """Test CaptionWorker.run emits progress signals."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (True, "OK")
        generator.generate_florence.return_value = "test caption"

        worker = CaptionWorker(
            generator=generator, image_path="/fake/path.jpg", backend="florence2"
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify progress signals were emitted
        progress_calls = [call[0][0] for call in worker.progress.emit.call_args_list]
        assert "Validating image..." in progress_calls
        assert "Loading Florence2 model..." in progress_calls

    def test_run_with_custom_florence_model(self):
        """Test CaptionWorker.run with custom florence_model."""
        generator = MagicMock(spec=CaptionGenerator)
        generator.validate_image.return_value = (True, "OK")
        generator.generate_florence.return_value = "test caption"

        worker = CaptionWorker(
            generator=generator,
            image_path="/fake/path.jpg",
            backend="florence2",
            florence_model="large",
        )

        # Mock signals
        worker.finished = MagicMock()
        worker.error = MagicMock()
        worker.progress = MagicMock()

        worker.run()

        # Verify generate_florence was called with correct model_key
        generator.generate_florence.assert_called_once_with(
            "/fake/path.jpg", model_key="large", detail_level="detailed"
        )
        worker.finished.emit.assert_called_once_with("test caption")
