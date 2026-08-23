"""Tests for get_caption_generator singleton behavior."""

import threading

from caption_generator import get_caption_generator, CaptionGenerator


class TestGetCaptionGenerator:
    """Test suite for get_caption_generator singleton behavior."""

    def test_get_caption_generator_returns_instance(self):
        """Test that get_caption_generator returns CaptionGenerator instance."""
        result = get_caption_generator()
        assert isinstance(result, CaptionGenerator)

    def test_get_caption_generator_singleton(self):
        """Test that get_caption_generator returns same instance."""
        instance1 = get_caption_generator()
        instance2 = get_caption_generator()
        assert instance1 is instance2

    def test_get_caption_generator_initialization(self):
        """Test that the returned instance has all expected attributes."""
        instance = get_caption_generator()
        assert hasattr(instance, "florence_model")
        assert hasattr(instance, "wd14_model")
        assert instance.florence_model is None
        assert instance.wd14_model is None

    def test_get_caption_generator_no_side_effects(self):
        """Test that multiple calls don't create additional instances."""
        instances = [get_caption_generator() for _ in range(10)]
        assert all(instance is instances[0] for instance in instances)

    def test_get_caption_generator_thread_safety(self):
        """Test that get_caption_generator is thread safe."""
        instances = []
        threads = []

        def get_instance():
            instances.append(get_caption_generator())

        for _ in range(10):
            thread = threading.Thread(target=get_instance)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert all(instance is instances[0] for instance in instances)

    def test_get_caption_generator_global_state(self):
        """Test that the function uses a global variable _generator_instance."""
        # This is implicitly tested by the singleton behavior
        # We can verify by checking the module has this attribute
        import caption_generator

        assert hasattr(caption_generator, "_generator_instance")

    def test_get_caption_generator_initial_state(self):
        """Test that the returned instance has correct initial state."""
        instance = get_caption_generator()
        assert hasattr(instance, "_florence_loaded")
        assert hasattr(instance, "_wd14_loaded")
        assert instance._florence_loaded is False
        assert instance._wd14_loaded is False
