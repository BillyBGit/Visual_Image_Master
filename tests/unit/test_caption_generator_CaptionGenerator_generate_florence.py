"""Tests for CaptionGenerator.generate_florence() method."""

import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image
import pytest

from caption_generator import CaptionGenerator


class TestCaptionGeneratorGenerateFlorence:
    """Test suite for CaptionGenerator.generate_florence method."""

    def test_generate_florence_with_valid_image(self):
        """Test generate_florence with a valid image."""
        # Create a temporary valid image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Create a simple RGB image
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Mock the Florence model and processor
            with patch.object(
                CaptionGenerator, "validate_image", return_value=(True, "Valid")
            ):
                with patch.object(CaptionGenerator, "load_florence"):
                    with patch("caption_generator.Image") as mock_image:
                        mock_image.open.return_value = img

                        with patch("caption_generator.np") as mock_np:
                            mock_array = MagicMock()
                            mock_array.shape = (100, 100, 3)
                            mock_array.size = 30000
                            mock_np.array.return_value = mock_array

                            generator = CaptionGenerator()
                            generator.florence_processor = MagicMock()

                            # Create a proper mock for the processor return value
                            mock_inputs = {
                                "pixel_values": MagicMock(),
                                "input_ids": MagicMock(),
                            }
                            mock_inputs["pixel_values"].shape = (
                                1,
                                3,
                                100,
                                100,
                            )  # Batch, Channels, Height, Width
                            mock_inputs["pixel_values"].to.return_value = mock_inputs[
                                "pixel_values"
                            ]

                            generator.florence_processor.return_value = mock_inputs
                            generator.florence_processor.batch_decode.return_value = [
                                "test caption"
                            ]
                            generator.florence_processor.post_process_generation.return_value = {
                                "<MORE_DETAILED_CAPTION>": "test caption"
                            }

                            generator.florence_model = MagicMock()
                            generator.florence_model.device = "cpu"
                            generator.florence_model.dtype = "float32"
                            generator.florence_model.generate.return_value = MagicMock()

                            result = generator.generate_florence(
                                image_path=tmp_path,
                                model_key="base",
                                detail_level="detailed",
                            )

                            # Assert the result is a non-empty string
                            assert isinstance(result, str)
                            assert len(result) > 0
        finally:
            os.unlink(tmp_path)

    def test_generate_florence_with_invalid_image_path(self):
        """Test generate_florence with a non-existent image path."""
        generator = CaptionGenerator()

        with pytest.raises(RuntimeError) as exc_info:
            generator.generate_florence(image_path="nonexistent.jpg")

        assert "Image file not found" in str(exc_info.value)

    def test_generate_florence_with_empty_image(self):
        """Test generate_florence with an empty image file."""
        # Create an empty file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            generator = CaptionGenerator()

            with pytest.raises(RuntimeError) as exc_info:
                generator.generate_florence(image_path=tmp_path)

            assert "Image file is empty" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    def test_generate_florence_with_different_detail_levels(self):
        """Test generate_florence with different detail levels."""
        # Create a temporary valid image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Mock the Florence model and processor
            with patch.object(
                CaptionGenerator, "validate_image", return_value=(True, "Valid")
            ):
                with patch.object(CaptionGenerator, "load_florence"):
                    with patch("caption_generator.Image") as mock_image:
                        mock_image.open.return_value = img

                        with patch("caption_generator.np") as mock_np:
                            mock_array = MagicMock()
                            mock_array.shape = (100, 100, 3)
                            mock_array.size = 30000
                            mock_np.array.return_value = mock_array

                            generator = CaptionGenerator()
                            generator.florence_processor = MagicMock()

                            # Create a proper mock for the processor return value
                            mock_inputs = {
                                "pixel_values": MagicMock(),
                                "input_ids": MagicMock(),
                            }
                            mock_inputs["pixel_values"].shape = (
                                1,
                                3,
                                100,
                                100,
                            )  # Batch, Channels, Height, Width
                            mock_inputs["pixel_values"].to.return_value = mock_inputs[
                                "pixel_values"
                            ]

                            generator.florence_processor.return_value = mock_inputs
                            generator.florence_processor.batch_decode.return_value = [
                                "test caption"
                            ]
                            generator.florence_processor.post_process_generation.return_value = {
                                "<DETAILED_CAPTION>": "test caption"
                            }

                            generator.florence_model = MagicMock()
                            generator.florence_model.device = "cpu"
                            generator.florence_model.dtype = "float32"
                            generator.florence_model.generate.return_value = MagicMock()

                            result = generator.generate_florence(
                                image_path=tmp_path, detail_level="brief"
                            )

                            # Assert the result is a non-empty string
                            assert isinstance(result, str)
                            assert len(result) > 0
        finally:
            os.unlink(tmp_path)

    def test_generate_florence_with_custom_model_key(self):
        """Test generate_florence with a custom model key."""
        # Create a temporary valid image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="green")
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Mock the Florence model and processor
            with patch.object(
                CaptionGenerator, "validate_image", return_value=(True, "Valid")
            ):
                with patch.object(CaptionGenerator, "load_florence"):
                    with patch("caption_generator.Image") as mock_image:
                        mock_image.open.return_value = img

                        with patch("caption_generator.np") as mock_np:
                            mock_array = MagicMock()
                            mock_array.shape = (100, 100, 3)
                            mock_array.size = 30000
                            mock_np.array.return_value = mock_array

                            generator = CaptionGenerator()
                            generator.florence_processor = MagicMock()

                            # Create a proper mock for the processor return value
                            mock_inputs = {
                                "pixel_values": MagicMock(),
                                "input_ids": MagicMock(),
                            }
                            mock_inputs["pixel_values"].shape = (
                                1,
                                3,
                                100,
                                100,
                            )  # Batch, Channels, Height, Width
                            mock_inputs["pixel_values"].to.return_value = mock_inputs[
                                "pixel_values"
                            ]

                            generator.florence_processor.return_value = mock_inputs
                            generator.florence_processor.batch_decode.return_value = [
                                "test caption"
                            ]
                            generator.florence_processor.post_process_generation.return_value = {
                                "<MORE_DETAILED_CAPTION>": "test caption"
                            }

                            generator.florence_model = MagicMock()
                            generator.florence_model.device = "cpu"
                            generator.florence_model.dtype = "float32"
                            generator.florence_model.generate.return_value = MagicMock()

                            result = generator.generate_florence(
                                image_path=tmp_path, model_key="custom"
                            )

                            # Assert the result is a non-empty string
                            assert isinstance(result, str)
                            assert len(result) > 0
        finally:
            os.unlink(tmp_path)

    def test_generate_florence_with_invalid_image_format(self):
        """Test generate_florence with an invalid image format."""
        # Create a file with invalid format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp.write("Not an image")
            tmp_path = tmp.name

        try:
            generator = CaptionGenerator()

            with pytest.raises(RuntimeError) as exc_info:
                generator.generate_florence(image_path=tmp_path)

            assert "File is not a valid image" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    def test_generate_florence_with_corrupt_image(self):
        """Test generate_florence with a corrupt image file."""
        # Create a file with corrupt image data
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="wb") as tmp:
            tmp.write(b"corrupt data that is not a valid image")
            tmp_path = tmp.name

        try:
            generator = CaptionGenerator()

            with pytest.raises(RuntimeError) as exc_info:
                generator.generate_florence(image_path=tmp_path)

            assert "File is not a valid image" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
