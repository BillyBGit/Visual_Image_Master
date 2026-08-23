"""Tests for CaptionGenerator.generate_ollama_text method."""

import json
import pytest
from unittest.mock import patch, MagicMock
from caption_generator import CaptionGenerator


class TestCaptionGeneratorGenerateOllamaText:
    """Test suite for CaptionGenerator.generate_llama_text method."""

    def test_generate_ollama_text_success(self):
        """Test successful text generation from Ollama."""
        # Setup
        caption_generator = CaptionGenerator()
        mock_response = {"response": "Generated caption text"}

        # Mock the urllib request
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = json.dumps(mock_response).encode(
                "utf-8"
            )
            mock_response_obj.__enter__ = MagicMock(return_value=mock_response_obj)
            mock_response_obj.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response_obj

            # Execute
            result = caption_generator.generate_ollama_text(
                url="http://localhost:11434", model="llama2", prompt="test prompt"
            )

            # Verify
            assert result == "Generated caption text"

    def test_generate_ollama_text_connection_error(self):
        """Test connection error handling."""
        # Setup
        caption_generator = CaptionGenerator()

        # Mock the urllib error - need to import the actual error class
        from urllib.error import URLError

        with patch("urllib.request.urlopen", side_effect=URLError("Connection failed")):
            with pytest.raises(RuntimeError) as exc_info:
                caption_generator.generate_ollama_text(
                    url="http://invalid-url:99999", model="llama2", prompt="test prompt"
                )
            assert "Failed to connect to Ollama" in str(exc_info.value)

    def test_generate_ollama_text_invalid_response(self):
        """Test invalid JSON response handling."""
        # Setup
        caption_generator = CaptionGenerator()

        # Mock invalid JSON response
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = b"invalid json"
            mock_response_obj.__enter__ = MagicMock(return_value=mock_response_obj)
            mock_response_obj.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response_obj

            with pytest.raises(RuntimeError) as exc_info:
                caption_generator.generate_ollama_text(
                    url="http://localhost:11434", model="llama2", prompt="test prompt"
                )
            assert "Invalid response from Ollama API" in str(exc_info.value)

    def test_generate_ollama_text_empty_response(self):
        """Test empty response handling."""
        # Setup
        caption_generator = CaptionGenerator()
        mock_response = {"response": ""}

        # Mock the urllib request
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = json.dumps(mock_response).encode(
                "utf-8"
            )
            mock_response_obj.__enter__ = MagicMock(return_value=mock_response_obj)
            mock_response_obj.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response_obj

            # Execute
            result = caption_generator.generate_ollama_text(
                url="http://localhost:11434", model="llama2", prompt="test prompt"
            )

            # Verify
            assert result == ""

    def test_generate_ollama_text_url_trailing_slash(self):
        """Test URL with trailing slash is handled correctly."""
        # Setup
        caption_generator = CaptionGenerator()
        mock_response = {"response": "Generated caption text"}

        # Mock the urllib request
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = json.dumps(mock_response).encode(
                "utf-8"
            )
            mock_response_obj.__enter__ = MagicMock(return_value=mock_response_obj)
            mock_response_obj.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response_obj

            # Execute
            result = caption_generator.generate_ollama_text(
                url="http://localhost:11434/", model="llama2", prompt="test prompt"
            )

            # Verify - should work without double slashes
            assert result == "Generated caption text"

    def test_generate_ollama_text_json_payload(self):
        """Test JSON payload structure."""
        # Setup
        caption_generator = CaptionGenerator()
        mock_response = {"response": "Generated caption text"}
        captured_data = None

        # Mock the urllib request to capture the data
        def capture_request(*args, **kwargs):
            nonlocal captured_data
            # The data is in kwargs
            captured_data = kwargs.get("data", None)
            return MagicMock()

        # Mock the urllib request
        with (
            patch("urllib.request.Request", side_effect=capture_request),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = json.dumps(mock_response).encode(
                "utf-8"
            )
            mock_response_obj.__enter__ = MagicMock(return_value=mock_response_obj)
            mock_response_obj.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response_obj

            # Execute
            caption_generator.generate_ollama_text(
                url="http://localhost:11434", model="llama2", prompt="test prompt"
            )

            # Verify the request was created with correct payload
            assert captured_data is not None
            # Handle both bytes and string data
            if isinstance(captured_data, str):
                payload = json.loads(captured_data)
            else:
                payload = json.loads(captured_data.decode("utf-8"))
            assert payload["model"] == "llama2"
            assert payload["prompt"] == "test prompt"
            assert not payload["stream"]

    def test_generate_ollama_text_timeout(self):
        """Test timeout handling."""
        # Setup
        caption_generator = CaptionGenerator()

        # Mock timeout error - need to import the actual error class
        from urllib.error import URLError

        with patch("urllib.request.urlopen", side_effect=URLError("Timeout")):
            with pytest.raises(RuntimeError) as exc_info:
                caption_generator.generate_ollama_text(
                    url="http://localhost:11434", model="llama2", prompt="test prompt"
                )
            assert "Failed to connect to Ollama" in str(exc_info.value)
