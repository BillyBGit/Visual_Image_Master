"""Tests for AutoCaptionDialog._on_backend_changed method."""

from unittest.mock import MagicMock
from autocaption_dialog import AutoCaptionDialog


class TestAutoCaptionDialogBackendChanged:
    """Test suite for AutoCaptionDialog._on_backend_changed visibility logic."""

    def test_florence2_backend_shows_florence_widget(self):
        """Test florence2 backend shows florence_widget and hides others."""
        # Create AutoCaptionDialog with mock image paths
        dialog = AutoCaptionDialog(["mock_image.jpg"])

        # Mock the combo box and widgets
        dialog.backend_combo = MagicMock()
        dialog.backend_combo.currentData.return_value = "florence2"
        dialog.florence_widget = MagicMock()
        dialog.wd14_widget = MagicMock()
        dialog.ollama_widget = MagicMock()

        # Call the method
        dialog._on_backend_changed(0)

        # Verify visibility changes
        dialog.florence_widget.setVisible.assert_called_once_with(True)
        dialog.wd14_widget.setVisible.assert_called_once_with(False)
        dialog.ollama_widget.setVisible.assert_called_once_with(False)

    def test_wd14_backend_shows_wd14_widget(self):
        """Test wd14 backend shows wd14_widget and hides others."""
        # Create AutoCaptionDialog with mock image paths
        dialog = AutoCaptionDialog(["mock_image.jpg"])

        # Mock the combo box and widgets
        dialog.backend_combo = MagicMock()
        dialog.backend_combo.currentData.return_value = "wd14"
        dialog.florence_widget = MagicMock()
        dialog.wd14_widget = MagicMock()
        dialog.ollama_widget = MagicMock()

        # Call the method
        dialog._on_backend_changed(1)

        # Verify visibility changes
        dialog.florence_widget.setVisible.assert_called_once_with(False)
        dialog.wd14_widget.setVisible.assert_called_once_with(True)
        dialog.ollama_widget.setVisible.assert_called_once_with(False)

    def test_ollama_backend_shows_ollama_widget(self):
        """Test ollama backend shows ollama_widget and hides others."""
        # Create AutoCaptionDialog with mock image paths
        dialog = AutoCaptionDialog(["mock_image.jpg"])

        # Mock the combo box and widgets
        dialog.backend_combo = MagicMock()
        dialog.backend_combo.currentData.return_value = "ollama"
        dialog.florence_widget = MagicMock()
        dialog.wd14_widget = MagicMock()
        dialog.ollama_widget = MagicMock()

        # Call the method
        dialog._on_backend_changed(2)

        # Verify visibility changes
        dialog.florence_widget.setVisible.assert_called_once_with(False)
        dialog.wd14_widget.setVisible.assert_called_once_with(False)
        dialog.ollama_widget.setVisible.assert_called_once_with(True)

    def test_backend_changed_updates_visibility_correctly(self):
        """Test all three backends in sequence, verifying visibility changes."""
        # Create AutoCaptionDialog with mock image paths
        dialog = AutoCaptionDialog(["mock_image.jpg"])

        # Mock the combo box and widgets
        dialog.backend_combo = MagicMock()
        dialog.florence_widget = MagicMock()
        dialog.wd14_widget = MagicMock()
        dialog.ollama_widget = MagicMock()

        # Test florence2 backend
        dialog.backend_combo.currentData.return_value = "florence2"
        dialog._on_backend_changed(0)
        dialog.florence_widget.setVisible.assert_called_with(True)
        dialog.wd14_widget.setVisible.assert_called_with(False)
        dialog.ollama_widget.setVisible.assert_called_with(False)

        # Test wd14 backend
        dialog.backend_combo.currentData.return_value = "wd14"
        dialog._on_backend_changed(1)
        dialog.florence_widget.setVisible.assert_called_with(False)
        dialog.wd14_widget.setVisible.assert_called_with(True)
        dialog.ollama_widget.setVisible.assert_called_with(False)

        # Test ollama backend
        dialog.backend_combo.currentData.return_value = "ollama"
        dialog._on_backend_changed(2)
        dialog.florence_widget.setVisible.assert_called_with(False)
        dialog.wd14_widget.setVisible.assert_called_with(False)
        dialog.ollama_widget.setVisible.assert_called_with(True)

    def test_initial_state_all_hidden(self):
        """Test all backend-specific widgets are hidden initially."""
        # Create AutoCaptionDialog with mock image paths
        dialog = AutoCaptionDialog(["mock_image.jpg"])

        # Mock the combo box and widgets
        dialog.backend_combo = MagicMock()
        dialog.backend_combo.currentData.return_value = None  # No backend selected
        dialog.florence_widget = MagicMock()
        dialog.wd14_widget = MagicMock()
        dialog.ollama_widget = MagicMock()

        # Call the method to set initial state
        dialog._on_backend_changed(0)

        # Verify initial state (should be hidden when no backend matches)
        dialog.florence_widget.setVisible.assert_called_with(False)
        dialog.wd14_widget.setVisible.assert_called_with(False)
        dialog.ollama_widget.setVisible.assert_called_with(False)
