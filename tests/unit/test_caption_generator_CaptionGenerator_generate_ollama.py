import json
import base64
from unittest.mock import patch, mock_open, MagicMock
import pytest

from caption_generator import CaptionGenerator


class TestCaptionGeneratorGenerateOllama:
    """Tests for CaptionGenerator.generate_ollama() method."""

    def test_generate_ollama_success(self):
        """Test successful caption generation."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Mock the HTTP request and response
        mock_response_content = json.dumps({"response": "Test caption"}).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_content
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("builtins.open", mock_open(read_data=b"test image data")):
                # Call the function
                result = caption_generator.generate_ollama(
                    image_path="test.jpg", url="http://localhost:11434", model="llava"
                )

                # Assert the function returns a non-empty caption
                assert result == "Test caption"

    def test_generate_ollama_custom_prompt(self):
        """Test caption generation with custom prompt."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Mock the HTTP request
        mock_response_content = json.dumps({"response": "Custom caption"}).encode(
            "utf-8"
        )
        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_content
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            with patch("builtins.open", mock_open(read_data=b"test image data")):
                # Call the function with custom prompt
                caption_generator.generate_ollama(
                    image_path="test.jpg",
                    url="http://localhost:11434",
                    model="llava",
                    custom_prompt="Custom prompt",
                )

                # Get the request that was made
                call_args = mock_urlopen.call_args
                request_data = json.loads(call_args[0][0].data.decode("utf-8"))

                # Assert the custom prompt is used in the request
                assert request_data["prompt"] == "Custom prompt"

    def test_generate_ollama_url_error(self):
        """Test handling of URL errors."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Mock urllib.error.URLError
        from urllib.error import URLError

        with patch("urllib.request.urlopen", side_effect=URLError("Connection failed")):
            with patch("builtins.open", mock_open(read_data=b"test image data")):
                # Assert raises RuntimeError with 'Failed to connect to Ollama' message
                with pytest.raises(RuntimeError) as exc_info:
                    caption_generator.generate_ollama(
                        image_path="test.jpg",
                        url="http://localhost:11434",
                        model="llava",
                    )
                assert "Failed to connect to Ollama" in str(exc_info.value)

    def test_generate_ollama_json_error(self):
        """Test handling of invalid JSON response."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Mock HTTP response with invalid JSON
        mock_response = MagicMock()
        mock_response.read.return_value = b"invalid json"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("builtins.open", mock_open(read_data=b"test image data")):
                # Assert raises RuntimeError with 'Invalid response from Ollama API' message
                with pytest.raises(RuntimeError) as exc_info:
                    caption_generator.generate_ollama(
                        image_path="test.jpg",
                        url="http://localhost:11434",
                        model="llava",
                    )
                assert "Invalid response from Ollama API" in str(exc_info.value)

    def test_generate_ollama_image_not_found(self):
        """Test handling of non-existent image path."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Assert raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            caption_generator.generate_ollama(
                image_path="nonexistent.jpg",
                url="http://localhost:11434",
                model="llava",
            )

    def test_generate_ollama_base64_encoding(self):
        """Test that image data is base64 encoded in the request payload."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Mock the HTTP request
        mock_response_content = json.dumps({"response": "Test caption"}).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_content
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            with patch("builtins.open", mock_open(read_data=b"test image data")):
                # Call the function
                caption_generator.generate_ollama(
                    image_path="test.jpg", url="http://localhost:11434", model="llava"
                )

                # Get the request that was made
                call_args = mock_urlopen.call_args
                request_data = json.loads(call_args[0][0].data.decode("utf-8"))

                # Verify the image data is base64 encoded
                image_data = request_data["images"][0]
                decoded_data = base64.b64decode(image_data).decode("utf-8")
                assert decoded_data == "test image data"

    def test_generate_ollama_payload_structure(self):
        """Test that the payload contains required keys."""
        # Create CaptionGenerator instance
        caption_generator = CaptionGenerator()

        # Mock the HTTP request
        mock_response_content = json.dumps({"response": "Test caption"}).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_content
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            with patch("builtins.open", mock_open(read_data=b"test image data")):
                # Call the function
                caption_generator.generate_ollama(
                    image_path="test.jpg", url="http://localhost:11434", model="llava"
                )

                # Get the request that was made
                call_args = mock_urlopen.call_args
                request_data = json.loads(call_args[0][0].data.decode("utf-8"))

                # Verify the payload contains required keys
                assert "model" in request_data
                assert "prompt" in request_data
                assert "images" in request_data
                assert "stream" in request_data
                assert request_data["stream"] is False
