"""Tests for CaptionWindow._load_settings method."""

from unittest.mock import MagicMock
from caption_window import CaptionWindow


class TestCaptionWindowLoadSettings:
    """Test suite for CaptionWindow._load_settings method."""

    def test_load_settings_with_parent_app(self):
        """Test loading settings from parent app with all values set."""
        # Create mock parent app with all settings
        parent_app = MagicMock()
        parent_app.caption_backend = "florence2"
        parent_app.caption_florence_model = "large"
        parent_app.caption_florence_detail = "concise"
        parent_app.caption_florence_custom_path = "/path/to/model"
        parent_app.caption_ollama_url = "http://custom:11434"
        parent_app.caption_ollama_model = "custom-model"
        parent_app.caption_wd14_model = "convnext"
        parent_app.caption_wd14_threshold = 0.75
        parent_app.caption_wd14_custom_paths = {"custom": "/path/to/custom"}
        parent_app.caption_blend_instructions = "Blend instructions"
        parent_app.caption_blend_model = "blend-model"

        # Create CaptionWindow with parent app
        # Store parent_app reference before creating window
        self.parent_app = parent_app
        window = CaptionWindow([], 0, parent=None)
        # Set the parent_app after window creation
        window.parent_app = parent_app

        # Mock the combo boxes and widgets
        window.backend_combo = MagicMock()
        window.backend_combo.count.return_value = 3
        window.backend_combo.itemData.side_effect = ["florence2", "ollama", "wd14"]

        window.florence_model_combo = MagicMock()
        window.florence_model_combo.count.return_value = 2
        window.florence_model_combo.itemData.side_effect = ["base", "large"]

        window.florence_detail_combo = MagicMock()
        window.florence_detail_combo.count.return_value = 2
        window.florence_detail_combo.itemData.side_effect = ["detailed", "concise"]

        window.florence_custom_path = MagicMock()
        window.florence_custom_path.setText = MagicMock()

        window.ollama_url = MagicMock()
        window.ollama_url.setText = MagicMock()

        window.ollama_model = MagicMock()
        window.ollama_model.setText = MagicMock()

        window.wd14_model_combo = MagicMock()
        window.wd14_model_combo.count.return_value = 2
        window.wd14_model_combo.itemData.side_effect = ["vit-v2", "convnext"]

        window.wd14_threshold = MagicMock()
        window.wd14_threshold.setValue = MagicMock()

        window.generator = MagicMock()
        window.generator.set_wd14_custom_path = MagicMock()

        window.wd14_custom_path = MagicMock()
        window.wd14_custom_path.setText = MagicMock()

        window.blend_instructions = MagicMock()
        window.blend_instructions.setPlainText = MagicMock()

        # Call the method
        window._load_settings()

        # Verify backend selection
        assert window.backend_combo.setCurrentIndex.called
        # Verify Florence2 settings
        assert window.florence_model_combo.setCurrentIndex.called
        assert window.florence_detail_combo.setCurrentIndex.called
        assert window.florence_custom_path.setText.called_with("/path/to/model")
        # Verify Ollama settings
        assert window.ollama_url.setText.called_with("http://custom:11434")
        assert window.ollama_model.setText.called_with("custom-model")
        # Verify WD14 settings
        assert window.wd14_model_combo.setCurrentIndex.called
        assert window.wd14_threshold.setValue.called_with(0.75)
        assert window.generator.set_wd14_custom_path.called
        # Verify blend instructions
        assert window.blend_instructions.setPlainText.called_with("Blend instructions")

    def test_load_settings_without_parent_app(self):
        """Test loading settings without parent app."""
        # Create CaptionWindow without parent app
        window = CaptionWindow([], 0, parent=None)

        # Mock the widgets
        window.backend_combo = MagicMock()
        window.florence_model_combo = MagicMock()
        window.florence_detail_combo = MagicMock()
        window.florence_custom_path = MagicMock()
        window.ollama_url = MagicMock()
        window.ollama_model = MagicMock()
        window.wd14_model_combo = MagicMock()
        window.wd14_threshold = MagicMock()
        window.generator = MagicMock()
        window.wd14_custom_path = MagicMock()
        window.blend_instructions = MagicMock()

        # Call the method
        window._load_settings()

        # Verify no widgets were modified
        assert not window.backend_combo.setCurrentIndex.called
        assert not window.florence_model_combo.setCurrentIndex.called
        assert not window.florence_detail_combo.setCurrentIndex.called
        assert not window.florence_custom_path.setText.called
        assert not window.ollama_url.setText.called
        assert not window.ollama_model.setText.called
        assert not window.wd14_model_combo.setCurrentIndex.called
        assert not window.wd14_threshold.setValue.called
        assert not window.generator.set_wd14_custom_path.called
        assert not window.wd14_custom_path.setText.called
        assert not window.blend_instructions.setPlainText.called

    def test_load_settings_with_defaults(self):
        """Test loading settings with default values."""
        # Create mock parent app with no settings
        parent_app = MagicMock()

        # Create CaptionWindow with parent app
        window = CaptionWindow([], 0, parent=None)
        # Set the parent_app after window creation
        window.parent_app = parent_app

        # Mock the combo boxes and widgets
        window.backend_combo = MagicMock()
        window.backend_combo.count.return_value = 3
        window.backend_combo.itemData.side_effect = ["florence2", "ollama", "wd14"]

        window.florence_model_combo = MagicMock()
        window.florence_model_combo.count.return_value = 2
        window.florence_model_combo.itemData.side_effect = ["base", "large"]

        window.florence_detail_combo = MagicMock()
        window.florence_detail_combo.count.return_value = 2
        window.florence_detail_combo.itemData.side_effect = ["detailed", "concise"]

        window.florence_custom_path = MagicMock()
        window.florence_custom_path.setText = MagicMock()

        window.ollama_url = MagicMock()
        window.ollama_url.setText = MagicMock()

        window.ollama_model = MagicMock()
        window.ollama_model.setText = MagicMock()

        window.wd14_model_combo = MagicMock()
        window.wd14_model_combo.count.return_value = 2
        window.wd14_model_combo.itemData.side_effect = ["vit-v2", "convnext"]

        window.wd14_threshold = MagicMock()
        window.wd14_threshold.setValue = MagicMock()

        window.generator = MagicMock()
        window.generator.set_wd14_custom_path = MagicMock()

        window.wd14_custom_path = MagicMock()
        window.wd14_custom_path.setText = MagicMock()

        window.blend_instructions = MagicMock()
        window.blend_instructions.setPlainText = MagicMock()

        # Call the method
        window._load_settings()

        # Verify default values are set
        assert window.backend_combo.setCurrentIndex.called
        assert window.florence_model_combo.setCurrentIndex.called
        assert window.florence_detail_combo.setCurrentIndex.called
        assert window.ollama_url.setText.called_with("http://localhost:11434")
        assert window.ollama_model.setText.called_with("llava")
        assert window.wd14_model_combo.setCurrentIndex.called
        assert window.wd14_threshold.setValue.called_with(0.55)

    def test_load_settings_with_custom_wd14_path(self):
        """Test loading settings with custom WD14 paths."""
        # Create mock parent app with custom paths
        parent_app = MagicMock()
        parent_app.caption_wd14_model = "custom"
        parent_app.caption_wd14_custom_paths = {"custom": "/path/to/custom"}

        # Create CaptionWindow with parent app
        window = CaptionWindow([], 0, parent=None)
        # Set the parent_app after window creation
        window.parent_app = parent_app

        # Mock the widgets
        window.wd14_model_combo = MagicMock()
        window.wd14_model_combo.count.return_value = 3
        window.wd14_model_combo.itemData.side_effect = ["vit-v2", "convnext", "custom"]

        window.wd14_threshold = MagicMock()
        window.wd14_threshold.setValue = MagicMock()

        window.generator = MagicMock()
        window.generator.set_wd14_custom_path = MagicMock()

        window.wd14_custom_path = MagicMock()
        window.wd14_custom_path.setText = MagicMock()

        # Call the method
        window._load_settings()

        # Verify custom path is set
        assert window.generator.set_wd14_custom_path.called_with(
            "custom", "/path/to/custom"
        )
        assert window.wd14_custom_path.setText.called_with("/path/to/custom")

    def test_load_settings_with_empty_blend_instructions(self):
        """Test loading settings with empty blend instructions."""
        # Create mock parent app with empty blend instructions
        parent_app = MagicMock()
        parent_app.caption_blend_instructions = ""

        # Create CaptionWindow with parent app
        window = CaptionWindow([], 0, parent=None)
        # Set the parent_app after window creation
        window.parent_app = parent_app

        # Mock the widgets
        window.blend_instructions = MagicMock()
        window.blend_instructions.setPlainText = MagicMock()

        # Call the method
        window._load_settings()

        # Verify blend instructions widget was not modified
        assert not window.blend_instructions.setPlainText.called

    def test_load_settings_with_invalid_backend(self):
        """Test loading settings with invalid backend value."""
        # Create mock parent app with invalid backend
        parent_app = MagicMock()
        parent_app.caption_backend = "invalid_backend"

        # Create CaptionWindow with parent app
        window = CaptionWindow([], 0, parent=None)
        # Set the parent_app after window creation
        window.parent_app = parent_app

        # Mock the combo boxes
        window.backend_combo = MagicMock()
        window.backend_combo.count.return_value = 3
        window.backend_combo.itemData.side_effect = ["florence2", "ollama", "wd14"]

        # Call the method
        window._load_settings()

        # Verify no crash and backend combo not modified
        assert not window.backend_combo.setCurrentIndex.called

    def test_load_settings_with_invalid_model(self):
        """Test loading settings with invalid Florence2 model value."""
        # Create mock parent app with invalid model
        parent_app = MagicMock()
        parent_app.caption_florence_model = "invalid_model"

        # Create CaptionWindow with parent app
        window = CaptionWindow([], 0, parent=None)
        # Set the parent_app after window creation
        window.parent_app = parent_app

        # Mock the combo boxes
        window.florence_model_combo = MagicMock()
        window.florence_model_combo.count.return_value = 2
        window.florence_model_combo.itemData.side_effect = ["base", "large"]

        # Call the method
        window._load_settings()

        # Verify no crash and model combo not modified
        assert not window.florence_model_combo.setCurrentIndex.called
