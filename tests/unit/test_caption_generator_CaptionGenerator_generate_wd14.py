"""Tests for CaptionGenerator.generate_wd14 method."""

from unittest.mock import patch, MagicMock
from caption_generator import CaptionGenerator


class TestCaptionGeneratorGenerateWD14:
    """Test suite for CaptionGenerator.generate_wd14 method."""

    def test_generate_wd14_valid_image(self):
        """Test generate_wd14 with valid image and ONNX format."""
        generator = CaptionGenerator()

        mock_image = MagicMock()

        with patch.object(generator, "validate_image", return_value=(True, "Image OK")):
            with patch.object(generator, "load_wd14"):
                with patch.object(
                    generator, "_generate_wd14_onnx", return_value=[0.5, 0.6, 0.2]
                ):
                    with patch("PIL.Image.open", return_value=mock_image):
                        generator.wd14_tags = [
                            {"name": "cat", "category": 1},
                            {"name": "dog", "category": 1},
                            {"name": "car", "category": 2},
                        ]
                        generator.wd14_format = "onnx"
                        generator.wd14_img_size = 224
                        mock_image.convert.return_value = mock_image
                        mock_image.resize.return_value = mock_image

                        result = generator.generate_wd14("test.jpg", "vit-v2", 0.35)

                        assert result == "dog, cat"

    def test_generate_wd14_invalid_image(self):
        """Test generate_wd14 with invalid image raises RuntimeError."""
        generator = CaptionGenerator()

        with patch.object(
            generator, "validate_image", return_value=(False, "Image not found")
        ):
            try:
                generator.generate_wd14("invalid.jpg", "vit-v2", 0.35)
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Image validation failed" in str(e)

    def test_generate_wd14_threshold_filtering(self):
        """Test generate_wd14 filters tags based on threshold."""
        generator = CaptionGenerator()

        mock_image = MagicMock()

        with patch.object(generator, "validate_image", return_value=(True, "Image OK")):
            with patch.object(generator, "load_wd14"):
                with patch.object(
                    generator, "_generate_wd14_pytorch", return_value=[0.2, 0.8, 0.4]
                ):
                    with patch("PIL.Image.open", return_value=mock_image):
                        generator.wd14_tags = [
                            {"name": "cat", "category": 1},
                            {"name": "dog", "category": 1},
                            {"name": "car", "category": 2},
                        ]
                        generator.wd14_format = "pytorch"
                        generator.wd14_img_size = 224
                        mock_image.convert.return_value = mock_image
                        mock_image.resize.return_value = mock_image

                        result = generator.generate_wd14("test.jpg", "vit-v2", 0.5)

                        assert result == "dog"

    def test_generate_wd14_category_filtering(self):
        """Test generate_wd14 skips rating category tags."""
        generator = CaptionGenerator()

        mock_image = MagicMock()

        with patch.object(generator, "validate_image", return_value=(True, "Image OK")):
            with patch.object(generator, "load_wd14"):
                with patch.object(
                    generator, "_generate_wd14_onnx", return_value=[0.6, 0.7, 0.5]
                ):
                    with patch("PIL.Image.open", return_value=mock_image):
                        generator.wd14_tags = [
                            {"name": "cat", "category": 1},
                            {"name": "dog", "category": 1},
                            {"name": "car", "category": 9},
                        ]
                        generator.wd14_format = "onnx"
                        generator.wd14_img_size = 224
                        mock_image.convert.return_value = mock_image
                        mock_image.resize.return_value = mock_image

                        result = generator.generate_wd14("test.jpg", "vit-v2", 0.35)

                        assert result == "dog, cat"

    def test_generate_wd14_sorting(self):
        """Test generate_wd14 sorts tags by probability."""
        generator = CaptionGenerator()

        mock_image = MagicMock()

        with patch.object(generator, "validate_image", return_value=(True, "Image OK")):
            with patch.object(generator, "load_wd14"):
                with patch.object(
                    generator, "_generate_wd14_onnx", return_value=[0.7, 0.5, 0.8]
                ):
                    with patch("PIL.Image.open", return_value=mock_image):
                        generator.wd14_tags = [
                            {"name": "cat", "category": 1},
                            {"name": "dog", "category": 1},
                            {"name": "car", "category": 2},
                        ]
                        generator.wd14_format = "onnx"
                        generator.wd14_img_size = 224
                        mock_image.convert.return_value = mock_image
                        mock_image.resize.return_value = mock_image

                        result = generator.generate_wd14("test.jpg", "vit-v2", 0.35)

                        assert result == "car, cat, dog"

    def test_generate_wd14_underscore_replacement(self):
        """Test generate_wd14 replaces underscores with spaces."""
        generator = CaptionGenerator()

        mock_image = MagicMock()

        with patch.object(generator, "validate_image", return_value=(True, "Image OK")):
            with patch.object(generator, "load_wd14"):
                with patch.object(generator, "_generate_wd14_onnx", return_value=[0.6]):
                    with patch("PIL.Image.open", return_value=mock_image):
                        generator.wd14_tags = [{"name": "black_cat", "category": 1}]
                        generator.wd14_format = "onnx"
                        generator.wd14_img_size = 224
                        mock_image.convert.return_value = mock_image
                        mock_image.resize.return_value = mock_image

                        result = generator.generate_wd14("test.jpg", "vit-v2", 0.35)

                        assert result == "black cat"

    def test_generate_wd14_pytorch_path(self):
        """Test generate_wd14 uses PyTorch path when format is not ONNX."""
        generator = CaptionGenerator()

        mock_image = MagicMock()

        with patch.object(generator, "validate_image", return_value=(True, "Image OK")):
            with patch.object(generator, "load_wd14"):
                with patch.object(
                    generator, "_generate_wd14_pytorch", return_value=[0.6]
                ):
                    with patch("PIL.Image.open", return_value=mock_image):
                        generator.wd14_tags = [{"name": "cat", "category": 1}]
                        generator.wd14_format = "pytorch"
                        generator.wd14_img_size = 224
                        mock_image.convert.return_value = mock_image
                        mock_image.resize.return_value = mock_image

                        result = generator.generate_wd14("test.jpg", "vit-v2", 0.35)

                        assert result == "cat"
