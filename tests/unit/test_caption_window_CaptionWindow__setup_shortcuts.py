"""Tests for CaptionWindow._setup_shortcuts method."""

from PySide6.QtGui import QShortcut, QKeySequence
from unittest.mock import patch

from caption_window import CaptionWindow


class TestCaptionWindowSetupShortcuts:
    """Test suite for CaptionWindow._setup_shortcuts method."""

    def test_setup_shortcuts_creates_save_shortcut(self):
        """Test that self.shortcut_save is created and connected to _on_save_requested."""
        window = CaptionWindow([], 0)

        # Verify shortcut_save exists and is a QShortcut instance
        assert isinstance(window.shortcut_save, QShortcut)

        # Verify it's connected to _on_save_requested
        assert window.shortcut_save.activated is not None

    def test_setup_shortcuts_save_shortcut_key_sequence(self):
        """Test that the save shortcut uses the standard Save key sequence."""
        window = CaptionWindow([], 0)

        # Verify the shortcut's key sequence is QKeySequence.StandardKey.Save
        assert window.shortcut_save.key().matches(QKeySequence.StandardKey.Save)

    def test_setup_shortcuts_save_shortcut_parent(self):
        """Test that the shortcut's parent is the CaptionWindow instance."""
        window = CaptionWindow([], 0)

        # Verify the shortcut's parent is the window
        assert window.shortcut_save.parent() is window

    def test_setup_shortcuts_save_shortcut_activated_signal(self):
        """Test that activating the shortcut calls _on_save_requested."""
        window = CaptionWindow([], 0)

        # Mock the _on_save_requested method
        with patch.object(window, "_on_save_requested") as mock_save:
            # Trigger the shortcut
            window.shortcut_save.activated.emit()

            # Verify _on_save_requested was called
            mock_save.assert_called_once()

    def test_setup_shortcuts_no_other_shortcuts(self):
        """Test that only self.shortcut_save is created and no other shortcuts exist."""
        window = CaptionWindow([], 0)

        # Verify only shortcut_save exists
        assert hasattr(window, "shortcut_save")

        # Check that no other shortcut attributes exist
        # (filter out methods and private attributes)
        shortcut_attrs = [
            attr
            for attr in dir(window)
            if "shortcut" in attr.lower()
            and not attr.startswith("_")
            and not callable(getattr(window, attr))
        ]
        assert shortcut_attrs == ["shortcut_save"]
