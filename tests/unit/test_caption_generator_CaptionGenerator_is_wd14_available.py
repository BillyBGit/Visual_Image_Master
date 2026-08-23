"""Tests for CaptionGenerator.is_wd14_available() method."""

from unittest.mock import patch, MagicMock
from caption_generator import CaptionGenerator
import sys


class TestCaptionGeneratorIsWd14Available:
    """Test suite for CaptionGenerator.is_wd14_available() method."""

    def test_wd14_available_with_onnx_and_torch(self):
        """Test with both ONNX and PyTorch backends available."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name in ["onnxruntime", "torch", "huggingface_hub"]:
                    return MagicMock()
                return sys.modules.get(name, __import__(name, *args))

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            assert result == (True, "WD14 ready (ONNX, PyTorch)")

    def test_wd14_available_with_only_onnx(self):
        """Test with only ONNX backend available."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name == "onnxruntime":
                    return MagicMock()
                elif name == "torch":
                    raise ImportError()
                elif name == "huggingface_hub":
                    return MagicMock()
                return sys.modules.get(name, __import__(name, *args))

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            assert result == (True, "WD14 ready (ONNX)")

    def test_wd14_available_with_only_torch(self):
        """Test with only PyTorch backend available."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name == "torch":
                    return MagicMock()
                elif name == "onnxruntime":
                    raise ImportError()
                elif name == "huggingface_hub":
                    return MagicMock()
                return sys.modules.get(name, __import__(name, *args))

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            assert result == (True, "WD14 ready (PyTorch)")

    def test_wd14_available_without_backends(self):
        """Test with no backends available."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name in ["torch", "onnxruntime"]:
                    raise ImportError()
                elif name == "huggingface_hub":
                    return MagicMock()
                return sys.modules.get(name, __import__(name, *args))

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            expected_msg = "Missing model backend. Install onnxruntime or torch: pip install onnxruntime"
            assert result == (False, expected_msg)

    def test_wd14_available_without_huggingface_hub(self):
        """Test with huggingface_hub unavailable."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name == "huggingface_hub":
                    raise ImportError()
                elif name in ["torch", "onnxruntime"]:
                    raise ImportError()
                # For other imports, let them through normally
                return None

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            assert result == (
                False,
                "Missing: huggingface_hub. Install with: pip install huggingface-hub",
            )

    def test_wd14_available_with_all_dependencies(self):
        """Test with all dependencies available."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name in ["onnxruntime", "torch", "huggingface_hub"]:
                    return MagicMock()
                return sys.modules.get(name, __import__(name, *args))

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            assert result == (True, "WD14 ready (ONNX, PyTorch)")

    def test_wd14_available_return_type(self):
        """Test that the return type is Tuple[bool, str]."""
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args):
                if name in ["onnxruntime", "torch", "huggingface_hub"]:
                    return MagicMock()
                return sys.modules.get(name, __import__(name, *args))

            mock_import.side_effect = import_side_effect

            generator = CaptionGenerator()
            result = generator.is_wd14_available()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
