"""Tests for the _debug function in caption_generator.py."""

import os
from unittest.mock import patch

from caption_generator import _debug


class TestDebug:
    """Test suite for _debug function."""

    def test_debug_prints_when_enabled(self):
        """Test that _debug prints when IMG_SORT_DEBUG is set to '1'."""
        os.environ["IMG_SORT_DEBUG"] = "1"
        with patch("builtins.print") as mock_print:
            _debug("test message")
            mock_print.assert_called_once_with("[DEBUG] test message")

    def test_debug_silence_when_disabled(self):
        """Test _debug silence when IMG_SORT_DEBUG is set to '0'."""
        os.environ["IMG_SORT_DEBUG"] = "0"
        with patch("builtins.print") as mock_print:
            _debug("test message")
            mock_print.assert_not_called()

    def test_debug_silence_when_undefined(self):
        """Test that _debug does not print when IMG_SORT_DEBUG is not set."""
        if "IMG_SORT_DEBUG" in os.environ:
            del os.environ["IMG_SORT_DEBUG"]
        with patch("builtins.print") as mock_print:
            _debug("test message")
            mock_print.assert_not_called()

    def test_debug_with_empty_message(self):
        """Test that _debug handles empty messages correctly."""
        os.environ["IMG_SORT_DEBUG"] = "1"
        with patch("builtins.print") as mock_print:
            _debug("")
            mock_print.assert_called_once_with("[DEBUG] ")

    def test_debug_with_special_characters(self):
        """Test that _debug handles special characters in messages."""
        os.environ["IMG_SORT_DEBUG"] = "1"
        with patch("builtins.print") as mock_print:
            _debug("test\nmessage\twith\tspecial\nchars")
            mock_print.assert_called_once_with(
                "[DEBUG] test\nmessage\twith\tspecial\nchars"
            )
