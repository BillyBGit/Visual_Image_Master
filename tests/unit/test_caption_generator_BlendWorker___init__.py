"""Tests for BlendWorker.__init__ method."""

from PySide6.QtCore import QThread
from caption_generator import BlendWorker


class TestBlendWorkerInit:
    """Test suite for BlendWorker initialization."""

    def test_init_sets_generator(self):
        """Test that generator is set correctly."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert instance.generator is mock_generator

    def test_init_sets_ollama_url(self):
        """Test that ollama_url is set correctly."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert instance.ollama_url == "http://localhost:11434"

    def test_init_sets_model(self):
        """Test that model is set correctly."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert instance.model == "llava"

    def test_init_sets_instructions(self):
        """Test that instructions are set correctly."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert instance.instructions == "Combine these captions"

    def test_init_sets_generated_caption(self):
        """Test that generated_caption is set correctly."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert instance.generated_caption == "A cat sitting"

    def test_init_sets_saved_caption(self):
        """Test that saved_caption is set correctly."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert instance.saved_caption == "A black cat"

    def test_init_inherits_qthread(self):
        """Test that BlendWorker inherits from QThread."""
        mock_generator = object()
        instance = BlendWorker(
            generator=mock_generator,
            ollama_url="http://localhost:11434",
            model="llava",
            instructions="Combine these captions",
            generated_caption="A cat sitting",
            saved_caption="A black cat",
        )
        assert isinstance(instance, QThread)
