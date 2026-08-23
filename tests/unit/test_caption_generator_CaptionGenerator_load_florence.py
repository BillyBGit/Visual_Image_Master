"""Tests for CaptionGenerator.load_florence() method."""

import pytest
from unittest.mock import patch
from caption_generator import CaptionGenerator


class TestCaptionGeneratorLoadFlorence:
    """Test suite for CaptionGenerator.load_florence() method."""

    def test_load_florence_missing_dependencies(self):
        """Test that missing dependencies raise RuntimeError."""
        caption_gen = CaptionGenerator()
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with pytest.raises(RuntimeError, match="Failed to load Florence2"):
                caption_gen.load_florence(model_key="base")
