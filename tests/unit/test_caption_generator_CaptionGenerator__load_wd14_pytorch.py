"""Tests for CaptionGenerator._load_wd14_pytorch method."""

import tempfile
import os
from unittest.mock import patch, MagicMock
import torch
import pytest

from caption_generator import CaptionGenerator


class TestCaptionGeneratorLoadWD14PyTorch:
    """Test suite for CaptionGenerator._load_wd14_pytorch method."""

    def test_load_wd14_pytorch_with_safetensors(self):
        """Test loading WD14 model with SafeTensors format."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock wd14_tags
        caption_gen.wd14_tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]

        # Create a temporary SafeTensors file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_file = f.name

        try:
            # Mock the necessary components
            with (
                patch("safetensors.torch.load_file") as mock_load_file,
                patch("timm.create_model") as mock_create_model,
                patch.object(
                    caption_gen, "_infer_wd14_architecture_and_size"
                ) as mock_infer,
            ):
                # Setup mocks
                mock_load_file.return_value = {"key": torch.randn(10, 10)}
                mock_infer.return_value = ("vit_base_patch16_224", 448)
                mock_model = MagicMock()
                mock_create_model.return_value = mock_model

                # Call the method
                result = caption_gen._load_wd14_pytorch(
                    temp_file, model_key="vit-v2", format_type="safetensors"
                )

                # Assertions
                assert result == 448
                mock_load_file.assert_called_once_with(temp_file)
                mock_infer.assert_called_once()
                mock_create_model.assert_called_once()
                mock_model.load_state_dict.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_load_wd14_pytorch_with_pytorch_format(self):
        """Test loading WD14 model with PyTorch format."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock wd14_tags
        caption_gen.wd14_tags = [
            "tag1",
            "tag2",
            "tag3",
            "tag4",
            "tag5",
            "tag6",
            "tag7",
            "tag8",
            "tag9",
            "tag10",
        ]

        # Create a temporary PyTorch file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_file = f.name

        try:
            # Mock the necessary components
            with (
                patch("caption_generator._debug"),
                patch("torch.load") as mock_torch_load,
                patch("timm.create_model") as mock_create_model,
                patch.object(
                    caption_gen, "_infer_wd14_architecture_and_size"
                ) as mock_infer,
            ):
                # Setup mocks
                mock_torch_load.return_value = {"key": torch.randn(10, 10)}
                mock_infer.return_value = ("swinv2_base_window14_224", 448)
                mock_model = MagicMock()
                mock_create_model.return_value = mock_model

                # Call the method
                result = caption_gen._load_wd14_pytorch(
                    temp_file, model_key="swinv2-v2", format_type="pytorch"
                )

                # Assertions
                assert result == 448
                mock_torch_load.assert_called_once_with(
                    temp_file, map_location="cuda"
                )  # Assuming CUDA is available
                mock_infer.assert_called_once()
                mock_create_model.assert_called_once()
                mock_model.load_state_dict.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_load_wd14_pytorch_invalid_format(self):
        """Test loading WD14 model with invalid format."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_file = f.name

        try:
            # Mock the necessary components
            with patch("caption_generator._debug"):
                # Call the method with invalid format
                with pytest.raises(Exception):
                    caption_gen._load_wd14_pytorch(
                        temp_file, model_key="vit-v2", format_type="invalid"
                    )
        finally:
            os.unlink(temp_file)

    def test_load_wd14_pytorch_missing_file(self):
        """Test loading WD14 model with non-existent file."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Call the method with non-existent file
        with pytest.raises(Exception):
            caption_gen._load_wd14_pytorch(
                "/non/existent/file.safetensors",
                model_key="vit-v2",
                format_type="safetensors",
            )

    def test_load_wd14_pytorch_device_selection(self):
        """Test device selection when CUDA and MPS are not available."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock wd14_tags
        caption_gen.wd14_tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]

        # Create a temporary SafeTensors file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_file = f.name

        try:
            # Mock the necessary components
            with (
                patch("caption_generator._debug"),
                patch("safetensors.torch.load_file") as mock_load_file,
                patch("timm.create_model") as mock_create_model,
                patch.object(
                    caption_gen, "_infer_wd14_architecture_and_size"
                ) as mock_infer,
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                # Setup mocks
                mock_load_file.return_value = {"key": torch.randn(10, 10)}
                mock_infer.return_value = ("vit_base_patch16_224", 448)
                mock_model = MagicMock()
                mock_create_model.return_value = mock_model

                # Call the method
                result = caption_gen._load_wd14_pytorch(
                    temp_file, model_key="vit-v2", format_type="safetensors"
                )

                # Assertions
                assert result == 448
                mock_model.to.assert_called_once_with("cpu")
        finally:
            os.unlink(temp_file)

    def test_load_wd14_pytorch_architecture_inference(self):
        """Test architecture inference and image size return."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock wd14_tags
        caption_gen.wd14_tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]

        # Create a temporary SafeTensors file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_file = f.name

        try:
            # Mock the necessary components
            with (
                patch("caption_generator._debug"),
                patch("safetensors.torch.load_file") as mock_load_file,
                patch("timm.create_model") as mock_create_model,
                patch.object(
                    caption_gen, "_infer_wd14_architecture_and_size"
                ) as mock_infer,
            ):
                # Setup mocks
                mock_load_file.return_value = {"key": torch.randn(10, 10)}
                mock_infer.return_value = ("vit_base_patch16_224", 448)
                mock_model = MagicMock()
                mock_create_model.return_value = mock_model

                # Call the method
                result = caption_gen._load_wd14_pytorch(
                    temp_file, model_key="vit-v2", format_type="safetensors"
                )

                # Assertions
                assert result == 448
                mock_infer.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_load_wd14_pytorch_weight_loading(self):
        """Test that model weights are loaded correctly."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock wd14_tags
        caption_gen.wd14_tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]

        # Create a temporary SafeTensors file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            temp_file = f.name

        try:
            # Mock the necessary components
            with (
                patch("caption_generator._debug"),
                patch("safetensors.torch.load_file") as mock_load_file,
                patch("timm.create_model") as mock_create_model,
                patch.object(
                    caption_gen, "_infer_wd14_architecture_and_size"
                ) as mock_infer,
            ):
                # Setup mocks
                mock_state_dict = {
                    "layer1.weight": torch.randn(10, 10),
                    "layer1.bias": torch.randn(10),
                }
                mock_load_file.return_value = mock_state_dict
                mock_infer.return_value = ("vit_base_patch16_224", 448)
                mock_model = MagicMock()
                mock_create_model.return_value = mock_model

                # Call the method
                result = caption_gen._load_wd14_pytorch(
                    temp_file, model_key="vit-v2", format_type="safetensors"
                )

                # Assertions
                assert result == 448
                mock_model.load_state_dict.assert_called_once_with(
                    mock_state_dict, strict=False
                )
        finally:
            os.unlink(temp_file)
