"""Tests for AdvancedFilterDialog.browse_custom_model method."""

import pytest
from unittest.mock import patch
from advanced_filter_dialog import AdvancedFilterDialog


class TestAdvancedFilterDialogBrowseCustomModel:
    """Test suite for AdvancedFilterDialog.browse_custom_model method."""

    @pytest.fixture
    def dialog(self):
        """Create an AdvancedFilterDialog instance."""
        return AdvancedFilterDialog(parent=None)

    @pytest.fixture
    def mock_qfile_dialog(self):
        """Mock QFileDialog.getOpenFileName."""
        with patch("advanced_filter_dialog.QFileDialog.getOpenFileName") as mock:
            yield mock

    def test_browse_custom_model_updates_path_edit(self, dialog, mock_qfile_dialog):
        """Test that browse_custom_model updates custom_model_path_edit."""
        mock_qfile_dialog.return_value = (
            "/path/to/model.pt",
            "YOLO Models (*.pt *.pth *.safetensors *.weights *.onnx)",
        )
        dialog.browse_custom_model()
        assert dialog.custom_model_path_edit.text() == "/path/to/model.pt"

    def test_browse_custom_model_with_empty_path(self, dialog, mock_qfile_dialog):
        """Test that browse_custom_model handles empty path correctly."""
        mock_qfile_dialog.return_value = ("", "")
        dialog.browse_custom_model()
        assert dialog.custom_model_path_edit.text() == ""

    def test_browse_custom_model_with_different_file_types(
        self, dialog, mock_qfile_dialog
    ):
        """Test that browse_custom_model works with different file types."""
        mock_qfile_dialog.return_value = (
            "/path/to/model.onnx",
            "YOLO Models (*.pt *.pth *.safetensors *.weights *.onnx)",
        )
        dialog.browse_custom_model()
        assert dialog.custom_model_path_edit.text() == "/path/to/model.onnx"

    def test_browse_custom_model_with_all_files_filter(self, dialog, mock_qfile_dialog):
        """Test that browse_custom_model works with All Files filter."""
        mock_qfile_dialog.return_value = ("/path/to/model.weights", "All Files (*)")
        dialog.browse_custom_model()
        assert dialog.custom_model_path_edit.text() == "/path/to/model.weights"

    def test_browse_custom_model_updates_filter_settings(
        self, dialog, mock_qfile_dialog
    ):
        """Test that browse_custom_model updates filter_settings."""
        mock_qfile_dialog.return_value = (
            "/path/to/model.pt",
            "YOLO Models (*.pt *.pth *.safetensors *.weights *.onnx)",
        )
        dialog.browse_custom_model()
        assert dialog.filter_settings["custom_model_path"] == "/path/to/model.pt"

    def test_browse_custom_model_with_no_selection(self, dialog, mock_qfile_dialog):
        """Test that browse_custom_model handles no selection."""
        mock_qfile_dialog.return_value = ("", "")
        dialog.browse_custom_model()
        assert dialog.filter_settings["custom_model_path"] is None

    def test_browse_custom_model_with_invalid_path(self, dialog, mock_qfile_dialog):
        """Test that browse_custom_model accepts invalid paths."""
        mock_qfile_dialog.return_value = (
            "/invalid/path/model.pt",
            "YOLO Models (*.pt *.pth *.safetensors *.weights *.onnx)",
        )
        dialog.browse_custom_model()
        assert dialog.custom_model_path_edit.text() == "/invalid/path/model.pt"
