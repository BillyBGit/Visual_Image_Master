import os
import tempfile
from caption_generator import CaptionGenerator


class TestCaptionGeneratorSetWd14CustomPath:
    def test_set_valid_custom_path(self):
        """Test setting a valid custom path for a WD14 model."""
        cg = CaptionGenerator()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        try:
            cg.set_wd14_custom_path("vit-v2", temp_path)
            assert cg.wd14_custom_paths["vit-v2"] == temp_path
        finally:
            os.unlink(temp_path)

    def test_set_invalid_custom_path(self):
        """Test setting an invalid custom path for a WD14 model."""
        cg = CaptionGenerator()
        cg.set_wd14_custom_path("vit-v2", "/invalid/path/to/model.onnx")
        assert "vit-v2" not in cg.wd14_custom_paths

    def test_clear_existing_custom_path(self):
        """Test clearing an existing custom path by setting empty string."""
        cg = CaptionGenerator()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        try:
            cg.set_wd14_custom_path("vit-v2", temp_path)
            assert "vit-v2" in cg.wd14_custom_paths

            cg.set_wd14_custom_path("vit-v2", "")
            assert "vit-v2" not in cg.wd14_custom_paths
        finally:
            os.unlink(temp_path)

    def test_set_custom_path_for_different_model_keys(self):
        """Test setting custom paths for different model keys."""
        cg = CaptionGenerator()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f1:
            temp_path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f2:
            temp_path2 = f2.name

        try:
            cg.set_wd14_custom_path("vit-v2", temp_path1)
            cg.set_wd14_custom_path("swinv2-v2", temp_path2)
            assert cg.wd14_custom_paths["vit-v2"] == temp_path1
            assert cg.wd14_custom_paths["swinv2-v2"] == temp_path2
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)

    def test_set_custom_path_with_empty_string(self):
        """Test setting custom path with empty string."""
        cg = CaptionGenerator()
        cg.set_wd14_custom_path("vit-v2", "")
        assert "vit-v2" not in cg.wd14_custom_paths

    def test_set_custom_path_with_none(self):
        """Test setting custom path with None."""
        cg = CaptionGenerator()
        cg.set_wd14_custom_path("vit-v2", None)
        assert "vit-v2" not in cg.wd14_custom_paths

    def test_set_custom_path_with_non_existent_file(self):
        """Test setting custom path with non-existent file."""
        cg = CaptionGenerator()
        cg.set_wd14_custom_path("vit-v2", "/non/existent/path")
        assert "vit-v2" not in cg.wd14_custom_paths
