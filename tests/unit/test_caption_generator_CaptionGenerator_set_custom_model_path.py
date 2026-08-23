"""Tests for CaptionGenerator.set_custom_model_path method."""

from caption_generator import CaptionGenerator


class TestCaptionGeneratorSetCustomModelPath:
    """Test suite for CaptionGenerator.set_custom_model_path method."""

    def test_set_custom_model_path_with_valid_path(self):
        """Test setting a valid model path."""
        generator = CaptionGenerator()
        generator.set_custom_model_path("/path/to/model")
        assert generator.custom_model_path == "/path/to/model"

    def test_set_custom_model_path_with_empty_string(self):
        """Test setting an empty string should result in None."""
        generator = CaptionGenerator()
        generator.set_custom_model_path("")
        assert generator.custom_model_path is None

    def test_set_custom_model_path_with_none(self):
        """Test setting None should result in None."""
        generator = CaptionGenerator()
        generator.set_custom_model_path(None)
        assert generator.custom_model_path is None

    def test_set_custom_model_path_with_huggingface_id(self):
        """Test setting a HuggingFace model ID."""
        generator = CaptionGenerator()
        generator.set_custom_model_path("username/model-name")
        assert generator.custom_model_path == "username/model-name"

    def test_set_custom_model_path_preserves_existing_value(self):
        """Test that setting a new path replaces the old one."""
        generator = CaptionGenerator()
        generator.set_custom_model_path("/first/path")
        generator.set_custom_model_path("/second/path")
        assert generator.custom_model_path == "/second/path"

    def test_set_custom_model_path_with_special_characters(self):
        """Test setting a path with special characters."""
        generator = CaptionGenerator()
        generator.set_custom_model_path("/path/with spaces & symbols!")
        assert generator.custom_model_path == "/path/with spaces & symbols!"

    def test_set_custom_model_path_with_relative_path(self):
        """Test setting a relative path."""
        generator = CaptionGenerator()
        generator.set_custom_model_path("./relative/path")
        assert generator.custom_model_path == "./relative/path"
