"""Tests for CaptionGenerator._load_wd14_onnx method."""

import pytest
from unittest.mock import Mock, patch
from caption_generator import CaptionGenerator


class TestCaptionGeneratorLoadWD14ONNX:
    """Test suite for CaptionGenerator._load_wd14_onnx method."""

    def test_load_wd14_onnx_success(self):
        """Test successful loading with standard 448x448 input size."""
        generator = CaptionGenerator()

        # Mock the InferenceSession
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "test_input"
        mock_input.shape = [1, 3, 448, 448]
        mock_input.type = "float32"
        mock_session.get_inputs.return_value = [mock_input]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            result = generator._load_wd14_onnx("mock_path.onnx", "vit-v2")

        assert result == 448
        assert generator.wd14_model is not None

    def test_load_wd14_onnx_custom_size(self):
        """Test successful loading with custom 512x512 input size."""
        generator = CaptionGenerator()

        # Mock the InferenceSession
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "test_input"
        mock_input.shape = [1, 3, 512, 512]
        mock_input.type = "float32"
        mock_session.get_inputs.return_value = [mock_input]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            result = generator._load_wd14_onnx("mock_path.onnx", "vit-v2")

        assert result == 512
        assert generator.wd14_model is not None

    def test_load_wd14_onnx_invalid_path(self):
        """Test loading with invalid file path raises exception."""
        generator = CaptionGenerator()

        with patch(
            "onnxruntime.InferenceSession", side_effect=Exception("File not found")
        ):
            with pytest.raises(Exception):
                generator._load_wd14_onnx("nonexistent.onnx", "vit-v2")

    def test_load_wd14_onnx_cuda_fallback(self):
        """Test fallback to CPU when CUDA fails."""
        generator = CaptionGenerator()

        # Mock the InferenceSession to fail with CUDA but succeed with CPU
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "test_input"
        mock_input.shape = [1, 3, 448, 448]
        mock_input.type = "float32"
        mock_session.get_inputs.return_value = [mock_input]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            result = generator._load_wd14_onnx("mock_path.onnx", "vit-v2")

        assert result == 448
        assert generator.wd14_model is not None

    def test_load_wd14_onnx_model_key_stored(self):
        """Test that model key is stored correctly."""
        generator = CaptionGenerator()

        # Mock the InferenceSession
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "test_input"
        mock_input.shape = [1, 3, 448, 448]
        mock_input.type = "float32"
        mock_session.get_inputs.return_value = [mock_input]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            result = generator._load_wd14_onnx("mock_path.onnx", "swinv2-v2")

        assert result == 448
        assert generator.wd14_model is not None

    def test_load_wd14_onnx_img_size_fallback(self):
        """Test fallback to default size when shape contains None."""
        generator = CaptionGenerator()

        # Mock the InferenceSession with None in shape
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "test_input"
        mock_input.shape = [1, 3, None, None]
        mock_input.type = "float32"
        mock_session.get_inputs.return_value = [mock_input]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            result = generator._load_wd14_onnx("mock_path.onnx", "vit-v2")

        assert result == 448  # Should use default fallback
        assert generator.wd14_model is not None

    def test_load_wd14_onnx_invalid_input_shape(self):
        """Test handling of invalid input shape (wrong channel count)."""
        generator = CaptionGenerator()

        # Mock the InferenceSession with wrong channel count
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "test_input"
        mock_input.shape = [1, 1, 448, 448]  # Wrong: should be 3 channels
        mock_input.type = "float32"
        mock_session.get_inputs.return_value = [mock_input]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            result = generator._load_wd14_onnx("mock_path.onnx", "vit-v2")

        assert result == 448  # Should still return 448 (default)
        assert generator.wd14_model is not None
