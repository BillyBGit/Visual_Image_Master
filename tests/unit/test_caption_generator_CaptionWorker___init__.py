"""Tests for CaptionWorker.__init__ method."""

from caption_generator import CaptionWorker


class TestCaptionWorkerInit:
    """Test suite for CaptionWorker.__init__ method."""

    def test_init_with_florence_backend(self):
        """Create CaptionWorker with backend='florence2', verify all attributes are set correctly."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "florence2"

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
        )

        assert worker.generator == generator
        assert worker.image_path == image_path
        assert worker.backend == backend
        assert worker.ollama_url == ""
        assert worker.ollama_model == ""
        assert worker.florence_model == "base"
        assert worker.florence_detail == "detailed"
        assert worker.wd14_model == "vit-v2"
        assert worker.wd14_threshold == 0.35

    def test_init_with_wd14_backend(self):
        """Create CaptionWorker with backend='wd14', verify all attributes are set correctly."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "wd14"

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
        )

        assert worker.generator == generator
        assert worker.image_path == image_path
        assert worker.backend == backend
        assert worker.ollama_url == ""
        assert worker.ollama_model == ""
        assert worker.florence_model == "base"
        assert worker.florence_detail == "detailed"
        assert worker.wd14_model == "vit-v2"
        assert worker.wd14_threshold == 0.35

    def test_init_with_ollama_backend(self):
        """Create CaptionWorker with backend='ollama', verify all attributes are set correctly."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "ollama"

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
        )

        assert worker.generator == generator
        assert worker.image_path == image_path
        assert worker.backend == backend
        assert worker.ollama_url == ""
        assert worker.ollama_model == ""
        assert worker.florence_model == "base"
        assert worker.florence_detail == "detailed"
        assert worker.wd14_model == "vit-v2"
        assert worker.wd14_threshold == 0.35

    def test_init_with_custom_ollama_url(self):
        """Create CaptionWorker with custom ollama_url, verify ollama_url attribute is set."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "ollama"
        ollama_url = "http://localhost:11434"

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
            ollama_url=ollama_url,
        )

        assert worker.ollama_url == ollama_url

    def test_init_with_custom_florence_model(self):
        """Create CaptionWorker with custom florence_model, verify florence_model attribute is set."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "florence2"
        florence_model = "large"

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
            florence_model=florence_model,
        )

        assert worker.florence_model == florence_model

    def test_init_with_custom_wd14_model(self):
        """Create CaptionWorker with custom wd14_model, verify wd14_model attribute is set."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "wd14"
        wd14_model = "convnext"

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
            wd14_model=wd14_model,
        )

        assert worker.wd14_model == wd14_model

    def test_init_with_custom_wd14_threshold(self):
        """Create CaptionWorker with custom wd14_threshold, verify wd14_threshold attribute is set."""
        generator = object()
        image_path = "/path/to/image.jpg"
        backend = "wd14"
        wd14_threshold = 0.5

        worker = CaptionWorker(
            generator=generator,
            image_path=image_path,
            backend=backend,
            wd14_threshold=wd14_threshold,
        )

        assert worker.wd14_threshold == wd14_threshold
