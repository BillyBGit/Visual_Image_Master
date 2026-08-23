"""Tests for BlendWorker.run method."""

import os
from unittest.mock import Mock
from caption_generator import BlendWorker, CaptionGenerator


class TestBlendWorkerRun:
    """Test suite for BlendWorker.run method."""

    def test_blendworker_run_success(self):
        """Test BlendWorker.run() success case - finished signal emitted."""
        # Create mock generator
        mock_generator = Mock(spec=CaptionGenerator)
        mock_generator.generate_ollama_text.return_value = "blended caption"

        # Create BlendWorker instance
        worker = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llama3",
            instructions="Blend these captions",
            generated_caption="A beautiful landscape",
            saved_caption="Mountain view",
        )

        # Track signal emissions
        emitted_result = []
        worker.finished.connect(lambda result: emitted_result.append(result))

        # Run the worker
        worker.run()

        # Assertions
        assert len(emitted_result) == 1
        assert emitted_result[0] == "blended caption"
        mock_generator.generate_ollama_text.assert_called_once()

    def test_blendworker_run_error(self):
        """Test BlendWorker.run() error case - error signal emitted."""
        # Create mock generator that raises exception
        mock_generator = Mock(spec=CaptionGenerator)
        mock_generator.generate_ollama_text.side_effect = Exception("Test error")

        # Create BlendWorker instance
        worker = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llama3",
            instructions="Blend these captions",
            generated_caption="A beautiful landscape",
            saved_caption="Mountain view",
        )

        # Track signal emissions
        emitted_error = []
        worker.error.connect(lambda error: emitted_error.append(error))

        # Run the worker
        worker.run()

        # Assertions
        assert len(emitted_error) == 1
        assert "Test error" in emitted_error[0]

    def test_blendworker_debug_mode(self):
        """Test BlendWorker.run() with debug mode enabled."""
        # Set debug environment variable
        os.environ["IMG_SORT_DEBUG"] = "1"

        # Create mock generator
        mock_generator = Mock(spec=CaptionGenerator)
        mock_generator.generate_ollama_text.return_value = "blended caption"

        # Create BlendWorker instance
        worker = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llama3",
            instructions="Blend these captions",
            generated_caption="A beautiful landscape",
            saved_caption="Mountain view",
        )

        # Track signal emissions
        emitted_result = []
        worker.finished.connect(lambda result: emitted_result.append(result))

        # Run the worker
        worker.run()

        # Assertions
        assert len(emitted_result) == 1
        mock_generator.generate_ollama_text.assert_called_once()

        # Clean up
        del os.environ["IMG_SORT_DEBUG"]

    def test_blendworker_prompt_construction(self):
        """Test BlendWorker.run() prompt construction."""
        # Create mock generator
        mock_generator = Mock(spec=CaptionGenerator)
        mock_generator.generate_ollama_text.return_value = "blended caption"

        # Create BlendWorker instance with specific captions
        instructions = "Blend these captions"
        generated_caption = "A beautiful landscape"
        saved_caption = "Mountain view"

        worker = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llama3",
            instructions=instructions,
            generated_caption=generated_caption,
            saved_caption=saved_caption,
        )

        # Track signal emissions
        emitted_result = []
        worker.finished.connect(lambda result: emitted_result.append(result))

        # Run the worker
        worker.run()

        # Verify the prompt was constructed correctly
        call_args = mock_generator.generate_ollama_text.call_args
        assert call_args is not None
        prompt = call_args[0][2]  # Third argument (prompt)

        expected_prompt = (
            f"{instructions}\n\n"
            f"Generated caption:\n{generated_caption}\n\n"
            f"Saved caption:\n{saved_caption}"
        )
        assert prompt == expected_prompt

    def test_blendworker_ollama_text_call(self):
        """Test BlendWorker.run() calls generate_ollama_text with correct arguments."""
        # Create mock generator
        mock_generator = Mock(spec=CaptionGenerator)
        mock_generator.generate_ollama_text.return_value = "blended caption"

        # Create BlendWorker instance
        worker = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llama3",
            instructions="Blend these captions",
            generated_caption="A beautiful landscape",
            saved_caption="Mountain view",
        )

        # Track signal emissions
        emitted_result = []
        worker.finished.connect(lambda result: emitted_result.append(result))

        # Run the worker
        worker.run()

        # Verify generate_ollama_text was called with correct arguments
        mock_generator.generate_ollama_text.assert_called_once_with(
            "http://localhost:11434",
            "llama3",
            "Blend these captions\n\nGenerated caption:\nA beautiful landscape\n\nSaved caption:\nMountain view",
        )
