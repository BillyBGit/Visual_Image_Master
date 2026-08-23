"""Tests for CaptionGenerator._generate_wd14_onnx method."""

import numpy as np
import pytest
from unittest.mock import MagicMock
from PIL import Image

from caption_generator import CaptionGenerator


class TestCaptionGeneratorGenerateWD14ONNX:
    """Test suite for CaptionGenerator._generate_wd14_onnx method."""

    def test_generate_wd14_onnx_valid_input(self):
        """Test with valid RGB PIL Image input."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.run.return_value = [np.random.randn(1, 425).astype(np.float32)]
        caption_gen.wd14_model = mock_model

        # Create a PIL Image
        image = Image.new("RGB", (224, 224), color="red")

        # Call the method
        result = caption_gen._generate_wd14_onnx(image)

        # Assert the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (425,)

    def test_generate_wd14_onnx_grayscale_input(self):
        """Test with grayscale PIL Image input - should fail due to shape mismatch."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]
        caption_gen.wd14_model = mock_model

        # Create a grayscale PIL Image
        image = Image.new("L", (224, 224), color=128)

        # Assert the function raises ValueError due to shape mismatch
        with pytest.raises(ValueError):
            caption_gen._generate_wd14_onnx(image)

    def test_generate_wd14_onnx_rgba_input(self):
        """Test with RGBA PIL Image input - should fail due to shape mismatch."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]
        caption_gen.wd14_model = mock_model

        # Create an RGBA PIL Image
        image = Image.new("RGBA", (224, 224), color=(255, 0, 0, 128))

        # Assert the function raises ValueError due to shape mismatch
        with pytest.raises(ValueError):
            caption_gen._generate_wd14_onnx(image)

    def test_generate_wd14_onnx_different_sizes(self):
        """Test with PIL Images of different sizes."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.run.return_value = [np.random.randn(1, 425).astype(np.float32)]
        caption_gen.wd14_model = mock_model

        # Test different sizes
        sizes = [(100, 100), (200, 200), (500, 500)]
        for size in sizes:
            image = Image.new("RGB", size, color="blue")
            result = caption_gen._generate_wd14_onnx(image)
            assert isinstance(result, np.ndarray)
            assert result.shape == (425,)

    def test_generate_wd14_onnx_model_run_failure(self):
        """Test when model.run raises an exception."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model to raise an exception
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.run.side_effect = RuntimeError("Model run failed")
        caption_gen.wd14_model = mock_model

        # Create a PIL Image
        image = Image.new("RGB", (224, 224), color="green")

        # Assert the function raises an exception
        with pytest.raises(RuntimeError):
            caption_gen._generate_wd14_onnx(image)

    def test_generate_wd14_onnx_normalization_correctness(self):
        """Test that input normalization is applied correctly."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]

        # Track the input that was passed to model.run
        captured_input = None

        def capture_run(*args, **kwargs):
            nonlocal captured_input
            captured_input = list(kwargs.values())[0]  # Get the input array
            return [np.random.randn(1, 425).astype(np.float32)]

        mock_model.run.side_effect = capture_run
        caption_gen.wd14_model = mock_model

        # Create a PIL Image with known pixel values
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Call the method
        caption_gen._generate_wd14_onnx(image)

        # Assert the input was normalized correctly
        assert captured_input is not None
        assert captured_input.shape[0] == 1  # batch dimension
        # For a gray pixel (128, 128, 128), after normalization it should be around 0
        # (128/255 - 0.485) / 0.229 ≈ (0.50196 - 0.485) / 0.229 ≈ 0.074
        expected_value = (128 / 255 - 0.485) / 0.229
        assert (
            abs(captured_input[0, 112, 112, 0] - expected_value) < 0.01
        )  # Check center pixel

    def test_generate_wd14_onnx_batch_dimension(self):
        """Test that the input has correct batch dimension."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the ONNX model
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "test_input"
        mock_model.get_inputs.return_value = [mock_input]

        # Track the input that was passed to model.run
        captured_input = None

        def capture_run(*args, **kwargs):
            nonlocal captured_input
            captured_input = kwargs[input_name]
            return [np.random.randn(1, 425).astype(np.float32)]

        input_name = "test_input"
        mock_model.run.side_effect = capture_run
        caption_gen.wd14_model = mock_model

        # Create a PIL Image
        image = Image.new("RGB", (224, 224), color="yellow")

        # Call the method
        caption_gen._generate_wd14_onnx(image)

        # Assert the input has correct batch dimension
        assert captured_input is not None
        assert captured_input.shape[0] == 1  # batch dimension
        assert len(captured_input.shape) == 4  # NHWC format
        assert captured_input.shape[3] == 3  # 3 channels (RGB)
