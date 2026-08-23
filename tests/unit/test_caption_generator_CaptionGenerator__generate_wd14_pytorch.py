"""Tests for CaptionGenerator._generate_wd14_pytorch method."""

import numpy as np
from unittest.mock import MagicMock
from PIL import Image

from caption_generator import CaptionGenerator


class TestCaptionGeneratorGenerateWD14PyTorch:
    """Test suite for CaptionGenerator._generate_wd14_pytorch method."""

    def test_generate_wd14_pytorch_valid_image(self):
        """Test with a valid RGB image."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the wd14_model
        caption_gen.wd14_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        caption_gen.wd14_model.parameters.return_value = iter([mock_param])

        # Create a mock output tensor
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Create a valid PIL Image
        image = Image.new("RGB", (224, 224), color="red")

        # Call the method
        result = caption_gen._generate_wd14_pytorch(image)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (527,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_generate_wd14_pytorch_grayscale_image(self):
        """Test with a grayscale image (should be converted to RGB)."""
        caption_gen = CaptionGenerator()

        # Mock the wd14_model
        caption_gen.wd14_model = MagicMock()
        caption_gen.wd14_model.parameters.return_value = [MagicMock()]
        caption_gen.wd14_model.parameters.return_value.__next__.return_value.device = (
            "cpu"
        )

        # Create a mock output tensor
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Create a grayscale PIL Image
        image = Image.new("L", (224, 224), color=128)

        # Call the method
        result = caption_gen._generate_wd14_pytorch(image)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (527,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_generate_wd14_pytorch_different_sizes(self):
        """Test with images of different sizes."""
        caption_gen = CaptionGenerator()

        # Mock the wd14_model
        caption_gen.wd14_model = MagicMock()
        caption_gen.wd14_model.parameters.return_value = [MagicMock()]
        caption_gen.wd14_model.parameters.return_value.__next__.return_value.device = (
            "cpu"
        )

        # Create a mock output tensor
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Test with different sizes
        sizes = [(112, 112), (448, 448)]
        for size in sizes:
            image = Image.new("RGB", size, color="blue")
            result = caption_gen._generate_wd14_pytorch(image)
            assert isinstance(result, np.ndarray)
            assert result.shape == (527,)

    def test_generate_wd14_pytorch_device_handling(self):
        """Test device handling (CPU/CUDA)."""
        caption_gen = CaptionGenerator()

        # Mock the wd14_model with CUDA device
        caption_gen.wd14_model = MagicMock()
        caption_gen.wd14_model.parameters.return_value = [MagicMock()]
        caption_gen.wd14_model.parameters.return_value.__next__.return_value.device = (
            "cuda"
        )

        # Create a mock output tensor
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Create a PIL Image
        image = Image.new("RGB", (224, 224), color="green")

        # Call the method
        result = caption_gen._generate_wd14_pytorch(image)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (527,)

    def test_generate_wd14_pytorch_normalization(self):
        """Test image normalization with correct mean and std."""
        caption_gen = CaptionGenerator()

        # Mock the wd14_model
        caption_gen.wd14_model = MagicMock()
        caption_gen.wd14_model.parameters.return_value = [MagicMock()]
        caption_gen.wd14_model.parameters.return_value.__next__.return_value.device = (
            "cpu"
        )

        # Create a mock output tensor
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Create a PIL Image with known pixel values
        image = Image.new("RGB", (224, 224), color=(100, 150, 200))

        # Call the method
        result = caption_gen._generate_wd14_pytorch(image)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (527,)

    def test_generate_wd14_pytorch_output_shape(self):
        """Test output shape is correct."""
        caption_gen = CaptionGenerator()

        # Mock the wd14_model to return outputs of shape (1, 527)
        caption_gen.wd14_model = MagicMock()
        caption_gen.wd14_model.parameters.return_value = [MagicMock()]
        caption_gen.wd14_model.parameters.return_value.__next__.return_value.device = (
            "cpu"
        )

        # Create a mock output tensor with specific shape
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Create a PIL Image
        image = Image.new("RGB", (224, 224), color="yellow")

        # Call the method
        result = caption_gen._generate_wd14_pytorch(image)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (527,)

    def test_generate_wd14_pytorch_sigmoid_activation(self):
        """Test that sigmoid activation is applied to outputs."""
        caption_gen = CaptionGenerator()

        # Mock the wd14_model to return raw logits
        caption_gen.wd14_model = MagicMock()
        caption_gen.wd14_model.parameters.return_value = [MagicMock()]
        caption_gen.wd14_model.parameters.return_value.__next__.return_value.device = (
            "cpu"
        )

        # Create a mock output tensor with logits (some positive, some negative)
        mock_output = MagicMock()
        mock_output.shape = (1, 527)
        caption_gen.wd14_model.return_value = mock_output

        # Create a PIL Image
        image = Image.new("RGB", (224, 224), color="purple")

        # Call the method
        result = caption_gen._generate_wd14_pytorch(image)

        # Assertions - sigmoid should map all values to [0, 1]
        assert isinstance(result, np.ndarray)
        assert result.shape == (527,)
        assert np.all(result >= 0) and np.all(result <= 1)
