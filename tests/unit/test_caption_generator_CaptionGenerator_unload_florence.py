from unittest.mock import patch, MagicMock
from caption_generator import CaptionGenerator


class TestCaptionGeneratorUnloadFlorence:
    """Tests for CaptionGenerator.unload_florence() method."""

    def test_unload_florence_when_model_not_loaded(self):
        """Test unload_florence when model is not loaded -
        should do nothing."""
        generator = CaptionGenerator()

        # Set some initial state
        generator._florence_loaded = False
        generator.florence_model = MagicMock()
        generator.florence_processor = MagicMock()
        generator._current_florence_model = "some_model"

        # Call the method
        generator.unload_florence()

        # Assert no changes to instance state
        assert generator._florence_loaded is False
        assert generator.florence_model is not None
        assert generator.florence_processor is not None
        assert generator._current_florence_model == "some_model"

    def test_unload_florence_when_model_loaded(self):
        """Test unload_florence when model is loaded -
        should clear all model attributes."""
        generator = CaptionGenerator()

        # Set up loaded state
        generator._florence_loaded = True
        generator.florence_model = MagicMock()
        generator.florence_processor = MagicMock()
        generator._current_florence_model = "test"

        # Call the method
        generator.unload_florence()

        # Assert all model-related attributes are None and _florence_loaded is False
        assert generator.florence_model is None
        assert generator.florence_processor is None
        assert generator._florence_loaded is False
        assert generator._current_florence_model is None

    def test_unload_florence_calls_gc_collect(self):
        """Test that unload_florence calls gc.collect
        when model is loaded."""
        generator = CaptionGenerator()
        generator._florence_loaded = True

        with patch("gc.collect") as mock_gc_collect:
            generator.unload_florence()

        # Assert gc.collect() was called
        mock_gc_collect.assert_called_once()

    def test_unload_florence_calls_cuda_empty_cache(self):
        """Test that unload_florence calls torch.cuda.empty_cache
        when CUDA is available."""
        generator = CaptionGenerator()
        generator._florence_loaded = True

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                generator.unload_florence()

        # Assert torch.cuda.empty_cache() was called
        mock_empty_cache.assert_called_once()

    def test_unload_florence_handles_cuda_exception(self):
        """Test that unload_florence handles exceptions
        from CUDA operations gracefully."""
        generator = CaptionGenerator()
        generator._florence_loaded = True

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache", side_effect=Exception("CUDA error")):
                # Should not raise exception
                generator.unload_florence()

    def test_unload_florence_clears_current_model_tracking(self):
        """Test that unload_florence clears _current_florence_model
        tracking."""
        generator = CaptionGenerator()
        generator._florence_loaded = True
        generator._current_florence_model = "test"

        generator.unload_florence()

        # Assert _current_florence_model is None
        assert generator._current_florence_model is None

    def test_unload_florence_calls_debug(self):
        """Test that unload_florence calls _debug with
        expected messages."""
        generator = CaptionGenerator()
        generator._florence_loaded = True

        with patch("caption_generator._debug") as mock_debug:
            generator.unload_florence()

        # Assert _debug() was called with expected messages
        debug_calls = [call[0] for call in mock_debug.call_args_list]
        assert any("Unloading Florence2 model" in str(call) for call in debug_calls)
        assert any("Florence2 model unloaded" in str(call) for call in debug_calls)
