"""Tests for CaptionGenerator.is_florence_available() method."""

import unittest
import unittest.mock as mock

from caption_generator import CaptionGenerator


class TestCaptionGeneratorIsFlorenceAvailable(unittest.TestCase):
    """Test suite for CaptionGenerator.is_florence_available() method."""

    def test_is_florence_available_with_torch_and_transformers(self):
        """Test that is_florence_available() returns (True, 'Florence2 ready') when torch and transformers are available."""
        # Mock successful imports
        with mock.patch.dict(
            "sys.modules", {"torch": mock.MagicMock(), "transformers": mock.MagicMock()}
        ):
            generator = CaptionGenerator()
            result = generator.is_florence_available()
            self.assertEqual(result, (True, "Florence2 ready"))

    def test_is_florence_available_with_missing_torch(self):
        """Test that is_florence_available() returns (False, message) when torch import fails."""
        # Mock torch import to fail
        with mock.patch.dict("sys.modules", {"torch": None}):
            generator = CaptionGenerator()
            result = generator.is_florence_available()
            self.assertEqual(result[0], False)
            self.assertIn("torch", result[1])

    def test_is_florence_available_with_missing_transformers(self):
        """Test that is_florence_available() returns (False, message) when transformers import fails."""
        # Mock transformers import to fail
        with mock.patch.dict("sys.modules", {"transformers": None}):
            generator = CaptionGenerator()
            result = generator.is_florence_available()
            self.assertEqual(result[0], False)
            self.assertIn("transformers", result[1])

    def test_is_florence_available_with_both_missing(self):
        """Test that is_florence_available() returns (False, message) when both torch and transformers are missing."""
        # Mock both imports to fail
        with mock.patch.dict("sys.modules", {"torch": None, "transformers": None}):
            generator = CaptionGenerator()
            result = generator.is_florence_available()
            self.assertEqual(result[0], False)
            # The message should contain information about missing dependencies
            self.assertTrue(
                "torch" in result[1] or "transformers" in result[1],
                f"Expected message to contain 'torch' or 'transformers', got: {result[1]}",
            )

    def test_is_florence_available_message_format(self):
        """Test that the message is a non-empty string."""
        # Mock successful imports to test message format
        with mock.patch.dict(
            "sys.modules", {"torch": mock.MagicMock(), "transformers": mock.MagicMock()}
        ):
            generator = CaptionGenerator()
            result = generator.is_florence_available()
            self.assertIsInstance(result[1], str)
            self.assertTrue(len(result[1]) > 0)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
