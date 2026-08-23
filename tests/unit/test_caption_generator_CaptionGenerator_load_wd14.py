"""Tests for CaptionGenerator.load_wd14 method."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest

from caption_generator import CaptionGenerator


class TestCaptionGeneratorLoadWD14:
    """Test suite for CaptionGenerator.load_wd14 method."""

    def test_load_wd14_default_model(self):
        """Test loading WD14 with default model (vit-v2)."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the dependencies to avoid actual downloads
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            # Setup mocks
            mock_download.return_value = "/tmp/mock_tags.csv"

            # Mock the ONNX loading
            with patch.object(caption_gen, "_load_wd14_onnx") as mock_load_onnx:
                mock_load_onnx.return_value = 448

                # Mock the import checking
                with patch("caption_generator.onnxruntime") as mock_onnx:
                    mock_onnx.__version__ = "1.0.0"

                    # Mock CSV
                    with patch("csv.DictReader") as mock_csv_reader:
                        mock_reader = MagicMock()
                        mock_reader.__iter__.return_value = [
                            {"name": "tag1", "category": "1"},
                            {"name": "tag2", "category": "2"},
                        ]
                        mock_csv_reader.return_value = mock_reader

                        # Call the method
                        result = caption_gen.load_wd14()

                        # Assertions
                        assert result is True
                        assert caption_gen._wd14_loaded is True
                        assert caption_gen.wd14_format == "onnx"

    def test_load_wd14_specific_model(self):
        """Test loading WD14 with specific model key."""
        caption_gen = CaptionGenerator()

        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.return_value = "/tmp/mock_tags.csv"

            with patch.object(caption_gen, "_load_wd14_onnx") as mock_load_onnx:
                mock_load_onnx.return_value = 448

                with patch("caption_generator.onnxruntime") as mock_onnx:
                    mock_onnx.__version__ = "1.0.0"

                    with patch("csv.DictReader") as mock_csv_reader:
                        mock_reader = MagicMock()
                        mock_reader.__iter__.return_value = [
                            {"name": "tag1", "category": "1"},
                            {"name": "tag2", "category": "2"},
                        ]
                        mock_csv_reader.return_value = mock_reader

                        result = caption_gen.load_wd14(model_key="vit-v2")

                        assert result is True
                        assert caption_gen._wd14_loaded is True

    def test_load_wd14_invalid_model_key(self):
        """Test loading WD14 with invalid model key."""
        caption_gen = CaptionGenerator()

        # Should fallback to vit-v2 and succeed, not raise error
        # Based on the implementation, invalid keys fallback to vit-v2
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.return_value = "/tmp/mock_tags.csv"

            with patch.object(caption_gen, "_load_wd14_onnx") as mock_load_onnx:
                mock_load_onnx.return_value = 448

                with patch("caption_generator.onnxruntime") as mock_onnx:
                    mock_onnx.__version__ = "1.0.0"

                    with patch("csv.DictReader") as mock_csv_reader:
                        mock_reader = MagicMock()
                        mock_reader.__iter__.return_value = [
                            {"name": "tag1", "category": "1"},
                            {"name": "tag2", "category": "2"},
                        ]
                        mock_csv_reader.return_value = mock_reader

                        # Should not raise, should fallback to vit-v2
                        result = caption_gen.load_wd14(model_key="invalid")
                        assert result is True

    def test_load_wd14_already_loaded(self):
        """Test that loading already loaded model returns True without reloading."""
        caption_gen = CaptionGenerator()

        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.return_value = "/tmp/mock_tags.csv"

            with patch.object(caption_gen, "_load_wd14_onnx") as mock_load_onnx:
                mock_load_onnx.return_value = 448

                with patch("caption_generator.onnxruntime") as mock_onnx:
                    mock_onnx.__version__ = "1.0.0"

                    with patch("csv.DictReader") as mock_csv_reader:
                        mock_reader = MagicMock()
                        mock_reader.__iter__.return_value = [
                            {"name": "tag1", "category": "1"},
                            {"name": "tag2", "category": "2"},
                        ]
                        mock_csv_reader.return_value = mock_reader

                        # First load
                        result1 = caption_gen.load_wd14()
                        assert result1 is True
                        assert caption_gen._wd14_loaded is True

                        # Second load - should return True without calling _load_wd14_onnx again
                        mock_load_onnx.reset_mock()
                        result2 = caption_gen.load_wd14()

                        assert result2 is True
                        assert (
                            mock_load_onnx.call_count == 0
                        )  # Should not be called again

    def test_load_wd14_different_model(self):
        """Test loading different WD14 models."""
        caption_gen = CaptionGenerator()

        with (
            patch("huggingface_hub.hf_hub_download") as mock_download,
            patch("caption_generator.onnxruntime") as mock_onnx,
            patch("csv") as mock_csv,
        ):
            mock_download.return_value = "/tmp/mock_tags.csv"
            mock_onnx.__version__ = "1.0.0"

            mock_reader = MagicMock()
            mock_reader.__enter__.return_value = mock_reader
            mock_reader.__exit__.return_value = None
            mock_reader.__iter__.return_value = [
                {"name": "tag1", "category": "1"},
                {"name": "tag2", "category": "2"},
            ]
            mock_csv.DictReader.return_value = mock_reader

            with patch.object(caption_gen, "_load_wd14_onnx") as mock_load_onnx:
                mock_load_onnx.return_value = 448

                # Load first model
                result1 = caption_gen.load_wd14(model_key="vit-v2")
                assert result1 is True

                # Load different model - should unload and reload
                mock_load_onnx.reset_mock()
                result2 = caption_gen.load_wd14(model_key="swinv2-v2")

                assert result2 is True
                assert mock_load_onnx.call_count == 1  # Should be called for new model

    def test_load_wd14_no_backend(self):
        """Test loading WD14 when no backend is available."""
        caption_gen = CaptionGenerator()

        # Mock missing dependencies
        with patch.dict("sys.modules", {"onnxruntime": None, "torch": None}):
            with pytest.raises(RuntimeError) as exc_info:
                caption_gen.load_wd14()

            assert "No model backend available" in str(exc_info.value)

    def test_load_wd14_custom_path(self):
        """Test loading WD14 with custom model path."""
        caption_gen = CaptionGenerator()

        # Create a temporary ONNX file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Set custom path
            caption_gen.set_wd14_custom_path("vit-v2", tmp_path)

            with (
                patch("huggingface_hub.hf_hub_download") as mock_download,
                patch("caption_generator.onnxruntime") as mock_onnx,
                patch("csv") as mock_csv,
            ):
                mock_download.return_value = "/tmp/mock_tags.csv"
                mock_onnx.__version__ = "1.0.0"

                mock_reader = MagicMock()
                mock_reader.__enter__.return_value = mock_reader
                mock_reader.__exit__.return_value = None
                mock_reader.__iter__.return_value = [
                    {"name": "tag1", "category": "1"},
                    {"name": "tag2", "category": "2"},
                ]
                mock_csv.DictReader.return_value = mock_reader

                with patch.object(caption_gen, "_load_wd14_onnx") as mock_load_onnx:
                    mock_load_onnx.return_value = 448

                    # Load should use custom path
                    result = caption_gen.load_wd14()

                    assert result is True
                    assert caption_gen._wd14_loaded is True

                    # Verify custom path was used
                    mock_load_onnx.assert_called_once()
                    args, kwargs = mock_load_onnx.call_args
                    assert args[0] == tmp_path

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
