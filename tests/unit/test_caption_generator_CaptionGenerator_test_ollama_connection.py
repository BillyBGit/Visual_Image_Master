import json
from unittest.mock import patch, MagicMock
import urllib.error
from caption_generator import CaptionGenerator


class TestCaptionGeneratorTestOllamaConnection:
    """Tests for CaptionGenerator.test_ollama_connection method."""

    def test_successful_connection_with_vision_models(self):
        """Test successful connection with vision models."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the API response
        mock_response = {"models": [{"name": "llava:latest"}]}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = str.encode(json.dumps(mock_response))
            mock_response_obj.decode.return_value = json.dumps(mock_response)
            mock_urlopen.return_value.__enter__.return_value = mock_response_obj

            # Call the method
            result = caption_gen.test_ollama_connection(url="http://localhost:11434")

            # Assert the result
            assert result[0] is True
            assert "Connected. Found 1 vision model(s)" in result[1]
            assert result[2] == ["llava:latest"]

    def test_successful_connection_without_vision_models(self):
        """Test successful connection without vision models."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the API response
        mock_response = {"models": [{"name": "llama2:latest"}]}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = str.encode(json.dumps(mock_response))
            mock_response_obj.decode.return_value = json.dumps(mock_response)
            mock_urlopen.return_value.__enter__.return_value = mock_response_obj

            # Call the method
            result = caption_gen.test_ollama_connection(url="http://localhost:11434")

            # Assert the result
            assert result[0] is True
            assert "Connected but no vision models found" in result[1]
            assert result[2] == ["llama2:latest"]

    def test_successful_connection_with_empty_models(self):
        """Test successful connection with empty models list."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the API response
        mock_response = {"models": []}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = str.encode(json.dumps(mock_response))
            mock_response_obj.decode.return_value = json.dumps(mock_response)
            mock_urlopen.return_value.__enter__.return_value = mock_response_obj

            # Call the method
            result = caption_gen.test_ollama_connection(url="http://localhost:11434")

            # Assert the result
            assert result[0] is True
            assert "Connected but no models installed" in result[1]
            assert result[2] == []

    def test_connection_failure_url_error(self):
        """Test connection failure with URL error."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            # Call the method
            result = caption_gen.test_ollama_connection(url="http://invalid:11434")

            # Assert the result
            assert result[0] is False
            assert "Connection failed" in result[1]
            assert result[2] == []

    def test_connection_timeout(self):
        """Test connection timeout."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("timed out")

            # Call the method
            result = caption_gen.test_ollama_connection(url="http://localhost:11434")

            # Assert the result
            assert result[0] is False
            assert "Connection failed" in result[1]
            assert result[2] == []

    def test_url_with_trailing_slash(self):
        """Test URL with trailing slash is properly stripped."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the API response
        mock_response = {"models": [{"name": "llava:latest"}]}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = str.encode(json.dumps(mock_response))
            mock_response_obj.decode.return_value = json.dumps(mock_response)
            mock_urlopen.return_value.__enter__.return_value = mock_response_obj

            # Call the method with trailing slash
            result = caption_gen.test_ollama_connection(url="http://localhost:11434/")

            # Assert the result
            assert result[0] is True
            # Verify the URL was called with trailing slash stripped
            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args[0][0]
            assert call_args.get_full_url() == ("http://localhost:11434/api/tags")

    def test_vision_model_detection_case_insensitive(self):
        """Test vision model detection is case insensitive."""
        # Create CaptionGenerator instance
        caption_gen = CaptionGenerator()

        # Mock the API response with mixed case models
        mock_response = {
            "models": [{"name": "Llava:latest"}, {"name": "Moondream:latest"}]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = str.encode(json.dumps(mock_response))
            mock_response_obj.decode.return_value = json.dumps(mock_response)
            mock_urlopen.return_value.__enter__.return_value = mock_response_obj

            # Call the method
            result = caption_gen.test_ollama_connection(url="http://localhost:11434")

            # Assert the result
            assert result[0] is True
            assert "Connected. Found 2 vision model(s)" in result[1]
            assert result[2] == ["Llava:latest", "Moondream:latest"]
