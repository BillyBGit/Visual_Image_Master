import os
import tempfile
from PIL import Image
from caption_generator import CaptionGenerator


class TestCaptionGeneratorValidateImage:
    def test_validate_image_nonexistent_file(self):
        """Test validation of non-existent file."""
        generator = CaptionGenerator()
        result = generator.validate_image("nonexistent.jpg")
        assert result == (False, "Image file not found: nonexistent.jpg")

    def test_validate_image_empty_file(self):
        """Test validation of empty file."""
        generator = CaptionGenerator()

        # Create temporary empty file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            empty_file_path = f.name

        try:
            result = generator.validate_image(empty_file_path)
            assert result == (False, "Image file is empty (0 bytes)")
        finally:
            os.unlink(empty_file_path)

    def test_validate_image_corrupt_file(self):
        """Test validation of corrupt image file."""
        generator = CaptionGenerator()

        # Create temporary file with invalid image data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            corrupt_file_path = f.name
            f.write(b"this is not a valid image")

        try:
            result = generator.validate_image(corrupt_file_path)
            assert result == (
                False,
                "File is not a valid image or format is not supported",
            )
        finally:
            os.unlink(corrupt_file_path)

    def test_validate_image_unsupported_mode(self):
        """Test validation of image with unsupported mode."""
        generator = CaptionGenerator()

        # Create temporary image with unsupported mode
        # (e.g., 'I' for 32-bit signed integer)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            unsupported_mode_file_path = f.name

            # Create a 100x100 image with 'I' mode
            img = Image.new("I", (100, 100), 128)
            img.save(unsupported_mode_file_path)

        try:
            result = generator.validate_image(unsupported_mode_file_path)
            # The actual mode when loaded will be 'I;16'
            # due to PNG conversion
            unsupported_mode = "I;16"
            assert result == (False, f"Unsupported image mode: {unsupported_mode}")
        finally:
            os.unlink(unsupported_mode_file_path)

    def test_validate_image_too_small(self):
        """Test validation of image that is too small."""
        generator = CaptionGenerator()

        # Create temporary 5x5 pixel image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            small_file_path = f.name
            img = Image.new("RGB", (5, 5), (255, 255, 255))
            img.save(small_file_path)

        try:
            result = generator.validate_image(small_file_path)
            assert result == (
                False,
                "Image too small: 5x5. Minimum 10x10 pixels required.",
            )
        finally:
            os.unlink(small_file_path)

    def test_validate_image_too_large(self):
        """Test validation of image that is too large."""
        generator = CaptionGenerator()

        # Create temporary 20000x20000 pixel image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            large_file_path = f.name
            # Create a small image but we'll test by creating a 10001x10001 image
            # which is just over the limit
            img = Image.new("RGB", (10001, 10001), (255, 255, 255))
            img.save(large_file_path)

        try:
            result = generator.validate_image(large_file_path)
            assert result == (
                False,
                "Image too large: 10001x10001. Maximum 10000x10000 pixels.",
            )
        finally:
            os.unlink(large_file_path)

    def test_validate_image_valid(self):
        """Test validation of valid image."""
        generator = CaptionGenerator()

        # Create temporary valid RGB image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            valid_file_path = f.name
            width, height = 100, 100
            img = Image.new("RGB", (width, height), (255, 255, 255))
            img.save(valid_file_path)

        try:
            result = generator.validate_image(valid_file_path)
            assert result[0] is True
            assert result[1] == f"Image OK ({width}x{height})"
        finally:
            os.unlink(valid_file_path)
