from unittest.mock import Mock, patch
from caption_generator import CaptionGenerator


class TestCaptionGeneratorUnloadWD14:
    """Tests for CaptionGenerator.unload_wd14 method."""

    def test_unload_wd14_when_not_loaded(self):
        """Test unload_wd14 when model is not loaded - should not raise exceptions."""
        generator = CaptionGenerator()

        # Should not raise any exceptions
        generator.unload_wd14()

        # _wd14_loaded should remain False
        assert not generator._wd14_loaded

    def test_unload_wd14_clears_model_state(self):
        """Test that unload_wd14 clears all model state."""
        generator = CaptionGenerator()

        # Set up initial state
        generator._wd14_loaded = True
        generator.wd14_model = "mock_model"
        generator.wd14_tags = ["tag1", "tag2"]
        generator.wd14_format = "onnx"
        generator._current_wd14_model = "vit-v2"
        generator.wd14_img_size = 512

        # Call unload
        generator.unload_wd14()

        # Verify all state is cleared
        assert generator.wd14_model is None
        assert generator.wd14_tags is None
        assert generator.wd14_format is None
        assert not generator._wd14_loaded
        assert generator._current_wd14_model is None
        # Reset to default
        assert generator.wd14_img_size == 448

    def test_unload_wd14_handles_onnx_session_deletion(self):
        """Test that unload_wd14 handles ONNX session deletion gracefully."""
        generator = CaptionGenerator()

        # Set up ONNX model state
        generator._wd14_loaded = True
        generator.wd14_format = "onnx"
        generator.wd14_model = Mock()  # Mock ONNX session

        # Should not raise exceptions
        generator.unload_wd14()

        # Verify model is cleared
        assert generator.wd14_model is None

    def test_unload_wd14_calls_gc_collect_twice(self):
        """Test that unload_wd14 calls gc.collect() twice."""
        generator = CaptionGenerator()
        generator._wd14_loaded = True

        with patch("gc.collect") as mock_gc_collect:
            generator.unload_wd14()

            # Verify gc.collect was called twice
            assert mock_gc_collect.call_count == 2

    def test_unload_wd14_clears_cuda_cache(self):
        """Test that unload_wd14 clears CUDA cache when using PyTorch."""
        generator = CaptionGenerator()
        generator._wd14_loaded = True

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                generator.unload_wd14()

                # Verify CUDA cache was cleared
                assert mock_empty_cache.call_count == 1

    def test_unload_wd14_sleeps_0_1_seconds(self):
        """Test that unload_wd14 sleeps for 0.1 seconds."""
        generator = CaptionGenerator()
        generator._wd14_loaded = True

        with patch("time.sleep") as mock_sleep:
            generator.unload_wd14()

            # Verify sleep was called with 0.1
            mock_sleep.assert_called_once_with(0.1)

    def test_unload_wd14_resets_wd14_img_size(self):
        """Test that unload_wd14 resets wd14_img_size to default."""
        generator = CaptionGenerator()
        generator._wd14_loaded = True
        generator.wd14_img_size = 512

        generator.unload_wd14()

        # Verify img_size is reset to default
        assert generator.wd14_img_size == 448
