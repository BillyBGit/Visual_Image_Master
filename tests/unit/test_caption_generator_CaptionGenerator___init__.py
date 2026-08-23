"""Tests for CaptionGenerator.__init__ method."""

from caption_generator import CaptionGenerator


class TestCaptionGeneratorInit:
    """Test suite for CaptionGenerator initialization."""

    def test_init_sets_florence_attributes_to_none(self):
        """Test that Florence attributes are initialized to None."""
        instance = CaptionGenerator()
        assert instance.florence_model is None
        assert instance.florence_processor is None
        assert instance.device is None
        assert instance._florence_loaded is False
        assert instance._current_florence_model is None
        assert instance.custom_model_path is None

    def test_init_sets_wd14_attributes_to_none(self):
        """Test that WD14 attributes are initialized to None."""
        instance = CaptionGenerator()
        assert instance.wd14_model is None
        assert instance.wd14_tags is None
        assert instance.wd14_format is None
        assert instance._wd14_loaded is False
        assert instance._current_wd14_model is None

    def test_init_sets_wd14_img_size_to_448(self):
        """Test that WD14 image size is initialized to 448."""
        instance = CaptionGenerator()
        assert instance.wd14_img_size == 448

    def test_init_sets_wd14_custom_paths_to_empty_dict(self):
        """Test that WD14 custom paths are initialized to empty dict."""
        instance = CaptionGenerator()
        assert instance.wd14_custom_paths == {}

    def test_init_creates_new_instance(self):
        """Test that CaptionGenerator() creates a new instance."""
        instance = CaptionGenerator()
        assert isinstance(instance, CaptionGenerator)

    def test_init_does_not_require_parameters(self):
        """Test that CaptionGenerator() can be called without parameters."""
        try:
            CaptionGenerator()
            assert True  # No exception raised
        except Exception:
            assert False, "CaptionGenerator() should not require parameters"

    def test_init_state_is_consistent(self):
        """Test that all attributes are set to their expected initial values."""
        instance = CaptionGenerator()

        # Florence attributes
        assert instance.florence_model is None
        assert instance.florence_processor is None
        assert instance.device is None
        assert instance._florence_loaded is False
        assert instance._current_florence_model is None
        assert instance.custom_model_path is None

        # WD14 attributes
        assert instance.wd14_model is None
        assert instance.wd14_tags is None
        assert instance.wd14_format is None
        assert instance.wd14_img_size == 448
        assert instance._wd14_loaded is False
        assert instance._current_wd14_model is None
        assert instance.wd14_custom_paths == {}
