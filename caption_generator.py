"""Caption generation backends for Florence2 and Ollama."""

import base64
import json
import os
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, QThread
from PIL import Image


def _debug(message: str):
    """Print debug message if debug mode is enabled via IMG_SORT_DEBUG env var."""
    if os.environ.get('IMG_SORT_DEBUG') == '1':
        print(f"[DEBUG] {message}")


# Florence2 model options with approximate sizes
FLORENCE_MODELS = {
    "base": ("florence-community/Florence-2-base", "~460MB"),
    "large": ("florence-community/Florence-2-large", "~1.5GB"),
    "base-ft": ("florence-community/Florence-2-base-ft", "~460MB"),
    "large-ft": ("florence-community/Florence-2-large-ft", "~1.5GB"),
}

# Florence2 detail level options (task prompts)
FLORENCE_DETAIL_LEVELS = {
    "brief": "<CAPTION>",
    "standard": "<DETAILED_CAPTION>",
    "detailed": "<MORE_DETAILED_CAPTION>",
}

# WD14 tagger model options with approximate sizes
WD14_MODELS = {
    "vit-v2": ("SmilingWolf/wd-v1-4-vit-tagger-v2", "~350MB"),
    "swinv2-v2": ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", "~430MB"),
    "convnextv2-v2": ("SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "~680MB"),
    "moat-v2": ("SmilingWolf/wd-v1-4-moat-tagger-v2", "~840MB"),
    "vit-v3": ("SmilingWolf/wd-vit-tagger-v3", "~350MB"),
}


class CaptionWorker(QThread):
    """Worker thread for caption generation to avoid blocking UI."""

    finished = Signal(str)  # Caption result
    error = Signal(str)  # Error message
    progress = Signal(str)  # Progress updates

    def __init__(self, generator, image_path: str, backend: str,
                 ollama_url: str = "", ollama_model: str = "",
                 florence_model: str = "base", florence_detail: str = "detailed",
                 wd14_model: str = "vit-v2", wd14_threshold: float = 0.35):
        super().__init__()
        self.generator = generator
        self.image_path = image_path
        self.backend = backend
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.florence_model = florence_model
        self.florence_detail = florence_detail
        self.wd14_model = wd14_model
        self.wd14_threshold = wd14_threshold

    def run(self):
        try:
            # Validate image first
            self.progress.emit("Validating image...")
            valid, message = self.generator.validate_image(self.image_path)
            if not valid:
                self.error.emit(f"Image validation failed: {message}")
                return

            if self.backend == "florence2":
                self.progress.emit("Loading Florence2 model...")
                caption = self.generator.generate_florence(
                    self.image_path,
                    model_key=self.florence_model,
                    detail_level=self.florence_detail
                )
            elif self.backend == "wd14":
                self.progress.emit("Loading WD14 tagger...")
                caption = self.generator.generate_wd14(
                    self.image_path,
                    model_key=self.wd14_model,
                    threshold=self.wd14_threshold
                )
            else:  # ollama
                self.progress.emit(f"Connecting to Ollama ({self.ollama_model})...")
                caption = self.generator.generate_ollama(
                    self.image_path, self.ollama_url, self.ollama_model
                )
            self.finished.emit(caption)
        except Exception as e:
            self.error.emit(str(e))


class BlendWorker(QThread):
    """Worker thread for caption blending to avoid blocking UI."""

    finished = Signal(str)  # Blended caption result
    error = Signal(str)  # Error message

    def __init__(self, generator, ollama_url: str, model: str,
                 instructions: str, generated_caption: str, saved_caption: str):
        super().__init__()
        self.generator = generator
        self.ollama_url = ollama_url
        self.model = model
        self.instructions = instructions
        self.generated_caption = generated_caption
        self.saved_caption = saved_caption

    def run(self):
        import os
        debug = os.environ.get('IMG_SORT_DEBUG') == '1'

        try:
            if debug:
                print(f"[DEBUG] BlendWorker starting with model: {self.model}")

            # Construct the prompt with both captions
            prompt = (
                f"{self.instructions}\n\n"
                f"Generated caption:\n{self.generated_caption}\n\n"
                f"Saved caption:\n{self.saved_caption}"
            )

            if debug:
                print(f"[DEBUG] BlendWorker prompt length: {len(prompt)}")

            # Call Ollama (text-only, no image)
            blended = self.generator.generate_ollama_text(
                self.ollama_url, self.model, prompt
            )

            if debug:
                print(f"[DEBUG] BlendWorker received response length: {len(blended)}")

            self.finished.emit(blended)

        except Exception as e:
            if debug:
                print(f"[DEBUG] BlendWorker error: {e}")
            self.error.emit(str(e))


class CaptionGenerator:
    """Generates image captions using Florence2, WD14, or Ollama."""

    def __init__(self):
        # Florence2 state
        self.florence_model = None
        self.florence_processor = None
        self.device = None
        self._florence_loaded = False
        self._current_florence_model = None  # Track which Florence model is loaded
        self.custom_model_path = None  # User-specified custom model path

        # WD14 state
        self.wd14_model = None
        self.wd14_tags = None  # Tag names from CSV
        self.wd14_format = None  # Track model format: "onnx", "safetensors", or "pytorch"
        self.wd14_img_size = 448  # Model input image size (detected from model)
        self._wd14_loaded = False
        self._current_wd14_model = None  # Track which WD14 model is loaded
        self.wd14_custom_paths = {}  # Dict mapping model_key to custom model file path

    def is_florence_available(self) -> Tuple[bool, str]:
        """Check if Florence2 dependencies are available.

        Returns:
            Tuple of (available, message)
        """
        try:
            import torch
            import transformers
            return True, "Florence2 ready"
        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else str(e)
            return False, f"Missing '{missing}' — install with: pip install transformers torch"

    def is_wd14_available(self) -> Tuple[bool, str]:
        """Check if WD14 dependencies are available.

        Returns:
            Tuple of (available, message)
        """
        # Check for at least one model loading backend
        has_onnx = False
        has_torch = False

        try:
            import onnxruntime
            has_onnx = True
        except ImportError:
            pass

        try:
            import torch
            has_torch = True
        except ImportError:
            pass

        # Need huggingface_hub for downloading
        try:
            import huggingface_hub
        except ImportError:
            return False, "Missing: huggingface_hub. Install with: pip install huggingface-hub"

        # Need at least one backend
        if not has_onnx and not has_torch:
            return False, "Missing model backend. Install onnxruntime or torch: pip install onnxruntime"

        backends = []
        if has_onnx:
            backends.append("ONNX")
        if has_torch:
            backends.append("PyTorch")

        return True, f"WD14 ready ({', '.join(backends)})"

    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        """Validate an image before caption generation.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (valid, message)
        """
        # Check file exists
        if not os.path.isfile(image_path):
            return False, f"Image file not found: {image_path}"

        # Check file size (reject empty files)
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return False, "Image file is empty (0 bytes)"

        try:
            # Try to open and validate the image
            with Image.open(image_path) as img:
                # Check format
                if img.format is None:
                    return False, "Unable to determine image format"

                # Check mode (must be convertible to RGB)
                valid_modes = ['RGB', 'RGBA', 'L', 'LA', 'P', '1', 'CMYK', 'YCbCr', 'LAB', 'HSV']
                if img.mode not in valid_modes:
                    return False, f"Unsupported image mode: {img.mode}"

                # Check dimensions
                width, height = img.size
                if width < 10 or height < 10:
                    return False, f"Image too small: {width}x{height}. Minimum 10x10 pixels required."

                if width > 10000 or height > 10000:
                    return False, f"Image too large: {width}x{height}. Maximum 10000x10000 pixels."

                # Try to actually load the image data to catch corrupt files
                try:
                    img.load()
                except Exception as e:
                    return False, f"Image data is corrupt: {e}"

                # Try converting to RGB to catch conversion issues
                try:
                    rgb_img = img.convert("RGB")
                    # Verify we got valid data
                    if rgb_img.size[0] <= 0 or rgb_img.size[1] <= 0:
                        return False, "Failed to convert image to RGB"
                except Exception as e:
                    return False, f"Cannot convert to RGB: {e}"

        except Image.UnidentifiedImageError:
            return False, "File is not a valid image or format is not supported"
        except Exception as e:
            return False, f"Error reading image: {e}"

        return True, f"Image OK ({width}x{height})"

    def set_custom_model_path(self, path: str):
        """Set a custom Florence2 model path (local directory or HuggingFace ID).

        Args:
            path: Path to local model directory or HuggingFace model ID
        """
        self.custom_model_path = path if path else None

    def set_wd14_custom_path(self, model_key: str, path: str):
        """Set a custom path for a specific WD14 model.

        Args:
            model_key: Key from WD14_MODELS dict (e.g., "vit-v2")
            path: Path to local model file (.onnx, .safetensors, .pt, etc.)
        """
        if path and os.path.isfile(path):
            self.wd14_custom_paths[model_key] = path
            _debug(f"Set custom path for WD14 model '{model_key}': {path}")
        elif model_key in self.wd14_custom_paths:
            # Clear custom path if empty path provided
            del self.wd14_custom_paths[model_key]
            _debug(f"Cleared custom path for WD14 model '{model_key}'")

    def unload_florence(self):
        """Unload the current Florence2 model to free memory."""
        if self._florence_loaded:
            _debug("Unloading Florence2 model...")
            import gc
            self.florence_model = None
            self.florence_processor = None
            self._florence_loaded = False
            self._current_florence_model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            _debug("Florence2 model unloaded")

    def load_florence(self, model_key: str = "base") -> bool:
        """Load Florence2 model (lazy loading on first use).

        Args:
            model_key: Key from FLORENCE_MODELS dict, or 'custom' to use custom_model_path

        Returns:
            True if loaded successfully
        """
        # Resolve model name from key or custom path
        if model_key == "custom" and self.custom_model_path:
            model_name = self.custom_model_path
        elif model_key in FLORENCE_MODELS:
            model_name = FLORENCE_MODELS[model_key][0]  # Tuple: (path, size)
        else:
            # Fallback to base model
            model_name = FLORENCE_MODELS["base"][0]

        # Check if same model is already loaded
        if self._florence_loaded and self._current_florence_model == model_name:
            _debug(f"Florence2 model '{model_name}' already loaded, skipping")
            return True

        # Unload current model if a different one is requested
        if self._florence_loaded and self._current_florence_model != model_name:
            _debug(f"Different model requested, unloading '{self._current_florence_model}'")
            self.unload_florence()

        try:
            import torch
            import transformers
            from transformers import AutoProcessor, Florence2ForConditionalGeneration

            _debug("=== Loading Florence2 ===")
            _debug(f"Model key: {model_key}")
            _debug(f"Model name: {model_name}")
            _debug(f"torch version: {torch.__version__}")
            _debug(f"transformers version: {transformers.__version__}")

            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            _debug(f"Using device: {self.device}")

            # Load processor
            _debug("Loading processor...")
            self.florence_processor = AutoProcessor.from_pretrained(model_name)
            _debug(f"Processor type: {type(self.florence_processor)}")
            _debug(f"Processor class: {self.florence_processor.__class__.__name__}")

            # Load model using native Florence2ForConditionalGeneration class
            _debug("Loading model (this may take a while)...")
            self.florence_model = Florence2ForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
            )
            _debug(f"Model type: {type(self.florence_model)}")
            _debug(f"Model class: {self.florence_model.__class__.__name__}")

            # Set to evaluation mode
            self.florence_model.eval()
            _debug("Model set to eval mode")

            self._florence_loaded = True
            self._current_florence_model = model_name
            _debug("=== Florence2 loaded successfully ===")
            return True

        except Exception as e:
            _debug(f"=== Florence2 load FAILED ===")
            _debug(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Florence2: {e}")

    def generate_florence(self, image_path: str, model_key: str = "base",
                          detail_level: str = "detailed") -> str:
        """Generate caption using Florence2.

        Args:
            image_path: Path to image file
            model_key: Key from FLORENCE_MODELS dict (base, large, base-ft, large-ft, custom)
            detail_level: Key from FLORENCE_DETAIL_LEVELS dict (brief, standard, detailed)

        Returns:
            Generated caption string
        """
        import torch

        _debug("=== Starting Florence2 caption generation ===")
        _debug(f"Image path: {image_path}")
        _debug(f"Model key: {model_key}, Detail level: {detail_level}")

        # Validate image first
        _debug("Validating image...")
        valid, message = self.validate_image(image_path)
        _debug(f"Validation result: valid={valid}, message={message}")
        if not valid:
            raise RuntimeError(f"Image validation failed: {message}")

        # Ensure model is loaded (will reload if different model requested)
        _debug("Ensuring model is loaded...")
        self.load_florence(model_key)

        # Load and preprocess image with explicit error handling
        _debug("Opening image with PIL...")
        try:
            image = Image.open(image_path)
            _debug(f"Image opened: format={image.format}, mode={image.mode}, size={image.size}")
            # Force load the image data
            image.load()
            _debug("Image data loaded")
            # Convert to RGB
            image = image.convert("RGB")
            _debug(f"Converted to RGB: mode={image.mode}, size={image.size}")
        except Exception as e:
            _debug(f"Failed to load image: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load image: {e}")

        # Store original size for post-processing
        original_size = (image.width, image.height)
        _debug(f"Original size: {original_size}")

        # Resize very large images to prevent memory issues
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            _debug(f"Resizing from {image.size} to {new_size}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Verify image has valid data by converting to numpy array
        _debug("Converting to numpy array...")
        try:
            img_array = np.array(image)
            _debug(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
            if img_array is None or img_array.size == 0:
                raise RuntimeError("Image converted to empty array")
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise RuntimeError(f"Invalid image shape: {img_array.shape}, expected (H, W, 3)")
            _debug("Array validation passed")
        except Exception as e:
            _debug(f"Array conversion failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Image array conversion failed: {e}")

        # Get task prompt based on detail level
        task_prompt = FLORENCE_DETAIL_LEVELS.get(detail_level, FLORENCE_DETAIL_LEVELS["detailed"])
        _debug(f"Task prompt: {task_prompt} (detail level: {detail_level})")

        # Process inputs - Florence2 expects specific format
        _debug("Calling florence_processor...")
        _debug(f"Processor type: {type(self.florence_processor)}")
        try:
            inputs = self.florence_processor(
                text=task_prompt,
                images=image,
                return_tensors="pt"
            )
            _debug(f"Processor returned: {type(inputs)}")
        except Exception as e:
            _debug(f"Processor call failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Florence2 processor failed: {e}")

        # Verify we have the expected inputs
        _debug(f"inputs is None: {inputs is None}")
        if inputs is None:
            raise RuntimeError("Florence2 processor returned None")

        _debug(f"inputs type: {type(inputs)}")
        _debug(f"inputs keys: {list(inputs.keys()) if hasattr(inputs, 'keys') else 'N/A'}")

        if "pixel_values" not in inputs:
            raise RuntimeError(f"Florence2 processor missing pixel_values. Got keys: {list(inputs.keys())}")

        _debug(f"pixel_values is None: {inputs['pixel_values'] is None}")
        if inputs["pixel_values"] is None:
            raise RuntimeError("Florence2 processor created None for pixel_values")

        # Verify pixel_values has valid shape
        pv = inputs["pixel_values"]
        _debug(f"pixel_values type: {type(pv)}")
        _debug(f"pixel_values has shape: {hasattr(pv, 'shape')}")
        if not hasattr(pv, 'shape'):
            raise RuntimeError(f"pixel_values has no shape attribute, type: {type(pv)}")
        _debug(f"pixel_values shape: {pv.shape}")
        if len(pv.shape) != 4:
            raise RuntimeError(f"Invalid pixel_values shape: {pv.shape}, expected 4D tensor")

        # Move inputs to model device, with pixel_values in model dtype
        _debug("Moving inputs to model device...")
        _debug(f"Model device: {self.florence_model.device}, dtype: {self.florence_model.dtype}")

        # Move each tensor appropriately - pixel_values needs float16, input_ids stays int
        inputs_dict = {}
        for key, value in inputs.items():
            if hasattr(value, 'to'):
                if key == "pixel_values":
                    # Convert pixel_values to model dtype (float16)
                    inputs_dict[key] = value.to(self.florence_model.device, self.florence_model.dtype)
                    _debug(f"{key} moved to {self.florence_model.device} with dtype {self.florence_model.dtype}")
                else:
                    # Keep other tensors (input_ids, attention_mask) as-is but move to device
                    inputs_dict[key] = value.to(self.florence_model.device)
                    _debug(f"{key} moved to {self.florence_model.device}")
            else:
                inputs_dict[key] = value

        # Generate using official HuggingFace pattern
        _debug("Starting model.generate()...")
        try:
            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    **inputs_dict,
                    max_new_tokens=1024,
                    num_beams=3,
                )
            _debug(f"generate() completed, output shape: {generated_ids.shape}")
        except Exception as e:
            _debug(f"model.generate() FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Decode using Florence2's post-processing
        _debug("Decoding output...")
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        _debug(f"Raw decoded text: {generated_text[:200]}...")

        # Use Florence2's built-in post-processing if available
        try:
            parsed = self.florence_processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=original_size
            )
            _debug(f"Post-processing result type: {type(parsed)}")
            # The result is a dict with the task as key
            caption = parsed.get(task_prompt, generated_text)
            _debug(f"Caption from post-processing: {caption[:200] if len(caption) > 200 else caption}")
        except Exception as e:
            _debug(f"Post-processing failed: {e}, using raw text")
            # Fallback: manual cleanup
            caption = generated_text

        # Clean up any remaining special tokens
        if isinstance(caption, str):
            caption = caption.replace(task_prompt, "")
            caption = caption.replace("</s>", "").replace("<s>", "")
            caption = caption.replace("<pad>", "").strip()

        _debug(f"=== Final caption: {caption[:200] if len(caption) > 200 else caption}")
        _debug("=== Florence2 generation complete ===")
        return caption

    def generate_ollama(self, image_path: str, url: str, model: str, custom_prompt: str = None) -> str:
        """Generate caption using Ollama API.

        Args:
            image_path: Path to image file
            url: Ollama API URL (e.g., http://localhost:11434)
            model: Model name (e.g., llava, bakllava)
            custom_prompt: Optional custom prompt (defaults to standard image description prompt)

        Returns:
            Generated caption string
        """
        import urllib.request
        import urllib.error

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Prepare request
        api_url = f"{url.rstrip('/')}/api/generate"

        # Use custom prompt if provided, otherwise use default
        if custom_prompt is None:
            custom_prompt = "Describe this image in detail. Focus on the main subjects, actions, setting, and any notable details."

        payload = {
            "model": model,
            "prompt": custom_prompt,
            "images": [image_data],
            "stream": False
        }

        request = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {e}")
        except json.JSONDecodeError:
            raise RuntimeError("Invalid response from Ollama API")

    def test_ollama_connection(self, url: str) -> Tuple[bool, str, list]:
        """Test connection to Ollama and get available models.

        Args:
            url: Ollama API URL

        Returns:
            Tuple of (connected, message, model_list)
        """
        import urllib.request
        import urllib.error

        api_url = f"{url.rstrip('/')}/api/tags"

        try:
            request = urllib.request.Request(api_url)
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in result.get("models", [])]
                # Filter to vision-capable models (common ones)
                vision_models = [m for m in models if any(
                    v in m.lower() for v in ["llava", "bakllava", "moondream", "llama3.2-vision"]
                )]
                if vision_models:
                    return True, f"Connected. Found {len(vision_models)} vision model(s)", vision_models
                elif models:
                    return True, f"Connected but no vision models found. Available: {', '.join(models[:5])}", models
                else:
                    return True, "Connected but no models installed", []
        except urllib.error.URLError as e:
            return False, f"Connection failed: {e}", []
        except Exception as e:
            return False, f"Error: {e}", []

    def generate_ollama_text(self, url: str, model: str, prompt: str) -> str:
        """Generate text response from Ollama without an image (text-only).

        Args:
            url: Ollama API URL
            model: Model name to use
            prompt: Text prompt to send

        Returns:
            Generated text response

        Raises:
            RuntimeError: If generation fails
        """
        import urllib.request
        import urllib.error

        api_url = f"{url.rstrip('/')}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        request = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {e}")
        except json.JSONDecodeError:
            raise RuntimeError("Invalid response from Ollama API")

    def unload_wd14(self):
        """Unload the current WD14 model to free memory."""
        if self._wd14_loaded:
            _debug(f"Unloading WD14 model (format: {self.wd14_format})...")
            import gc

            # Explicitly delete ONNX session if it exists
            if self.wd14_format == "onnx" and self.wd14_model is not None:
                try:
                    # ONNX Runtime sessions should be explicitly closed
                    del self.wd14_model
                    _debug("ONNX session deleted")
                except Exception as e:
                    _debug(f"Error deleting ONNX session: {e}")

            # Clear all model state
            self.wd14_model = None
            self.wd14_tags = None
            self.wd14_format = None
            self.wd14_img_size = 448  # Reset to default
            self._wd14_loaded = False
            self._current_wd14_model = None

            # Force garbage collection
            gc.collect()
            gc.collect()  # Run twice to catch cyclic references

            # Clear PyTorch cache if using PyTorch format
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    _debug("PyTorch CUDA cache cleared")
            except:
                pass

            # Give system time to release resources (helps prevent crashes when switching models)
            import time
            time.sleep(0.1)

            _debug("WD14 model unloaded")

    def load_wd14(self, model_key: str = "vit-v2") -> bool:
        """Load WD14 tagger model (lazy loading on first use).

        Automatically detects and loads the model in whatever format is available:
        - ONNX (.onnx) - preferred, fastest, no PyTorch needed
        - SafeTensors (.safetensors) - requires PyTorch/timm
        - PyTorch (.bin, .pt, .pth) - requires PyTorch/timm

        Args:
            model_key: Key from WD14_MODELS dict

        Returns:
            True if loaded successfully
        """
        import csv

        # Resolve model name from key
        if model_key in WD14_MODELS:
            model_name = WD14_MODELS[model_key][0]  # Tuple: (path, size)
        else:
            # Fallback to vit-v2
            model_name = WD14_MODELS["vit-v2"][0]

        # Check if same model is already loaded
        if self._wd14_loaded and self._current_wd14_model == model_name:
            _debug(f"WD14 model '{model_name}' already loaded, skipping")
            return True

        # Unload current model if a different one is requested
        if self._wd14_loaded and self._current_wd14_model != model_name:
            _debug(f"Different model requested, unloading '{self._current_wd14_model}'")
            _debug(f"Current model state before unload - loaded: {self._wd14_loaded}, format: {self.wd14_format}")
            self.unload_wd14()
            _debug("Unload complete, continuing with new model load")

        try:
            from huggingface_hub import hf_hub_download
            import shutil

            _debug("=== Loading WD14 ===")
            _debug(f"Model key: {model_key}")
            _debug(f"Model name: {model_name}")
            _debug(f"Current state - _wd14_loaded: {self._wd14_loaded}, model: {self.wd14_model}, tags: {self.wd14_tags}")

            # Download and load tags CSV
            _debug("Step 1: Downloading tags CSV...")
            try:
                tags_path = hf_hub_download(model_name, "selected_tags.csv")
                _debug(f"Tags CSV downloaded to: {tags_path}")
            except Exception as e:
                _debug(f"Failed to download tags CSV: {e}")
                raise

            _debug("Step 2: Loading tags from CSV...")
            try:
                with open(tags_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.wd14_tags = []
                    for row in reader:
                        self.wd14_tags.append({
                            'name': row['name'],
                            'category': int(row['category'])
                        })
                _debug(f"Loaded {len(self.wd14_tags)} tags")
            except Exception as e:
                _debug(f"Failed to load tags: {e}")
                raise

            # Check which backends are available
            _debug("Step 3: Checking available backends...")
            has_onnx = False
            has_torch = False
            try:
                import onnxruntime
                has_onnx = True
                _debug("ONNX Runtime available")
            except ImportError:
                _debug("ONNX Runtime not available")

            try:
                import torch
                has_torch = True
                _debug("PyTorch available")
            except ImportError:
                _debug("PyTorch not available")

            if not has_onnx and not has_torch:
                raise RuntimeError("No model backend available. Install onnxruntime or torch: pip install onnxruntime")

            # Extract model short name from repository (e.g., "SmilingWolf/wd-v1-4-vit-tagger-v2" -> "wd-v1-4-vit-tagger-v2")
            model_short_name = model_name.split('/')[-1]
            _debug(f"Model short name: {model_short_name}")

            # Check if user has specified a custom path for this model
            _debug("Step 4: Checking for custom model path...")
            model_path = None
            model_format = None

            if model_key in self.wd14_custom_paths:
                custom_path = self.wd14_custom_paths[model_key]
                if os.path.isfile(custom_path):
                    _debug(f"Using custom model path: {custom_path}")
                    model_path = custom_path
                    # Detect format from file extension
                    ext = os.path.splitext(custom_path)[1].lower()
                    if ext == ".onnx":
                        model_format = "onnx"
                    elif ext == ".safetensors":
                        model_format = "safetensors"
                    elif ext in [".pt", ".pth", ".bin"]:
                        model_format = "pytorch"
                    else:
                        _debug(f"Warning: Unknown file extension {ext}, will attempt to detect format")

            # If no custom path or custom path doesn't exist, try to download model in order of preference: ONNX -> SafeTensors -> PyTorch
            if not model_path:
                # Try ONNX format first (fastest, no PyTorch needed) - only if backend available
                if has_onnx:
                    try:
                        _debug("Attempting to download ONNX model...")
                        downloaded_path = hf_hub_download(model_name, "model.onnx")
                        # Check for renamed model first
                        renamed_path = os.path.join(os.path.dirname(downloaded_path), f"{model_short_name}.onnx")

                        # If renamed version already exists, use it
                        if os.path.exists(renamed_path):
                            model_path = renamed_path
                            _debug(f"Found existing renamed model: {renamed_path}")
                        # Otherwise, rename the downloaded file
                        elif downloaded_path != renamed_path:
                            shutil.copy2(downloaded_path, renamed_path)
                            _debug(f"Renamed model to: {renamed_path}")
                            model_path = renamed_path
                        else:
                            model_path = downloaded_path

                        model_format = "onnx"
                        _debug(f"Using ONNX model at: {model_path}")
                    except Exception as e:
                        _debug(f"ONNX model not found: {e}")
                else:
                    _debug("Skipping ONNX format (onnxruntime not installed)")

                # Try SafeTensors format - only if PyTorch backend available
                if not model_path and has_torch:
                    try:
                        _debug("Attempting to download SafeTensors model...")
                        downloaded_path = hf_hub_download(model_name, "model.safetensors")
                        # Check for renamed model first
                        renamed_path = os.path.join(os.path.dirname(downloaded_path), f"{model_short_name}.safetensors")

                        # If renamed version already exists, use it
                        if os.path.exists(renamed_path):
                            model_path = renamed_path
                            _debug(f"Found existing renamed model: {renamed_path}")
                        # Otherwise, rename the downloaded file
                        elif downloaded_path != renamed_path:
                            shutil.copy2(downloaded_path, renamed_path)
                            _debug(f"Renamed model to: {renamed_path}")
                            model_path = renamed_path
                        else:
                            model_path = downloaded_path

                        model_format = "safetensors"
                        _debug(f"Using SafeTensors model at: {model_path}")
                    except Exception as e:
                        _debug(f"SafeTensors model not found: {e}")
                elif not model_path:
                    _debug("Skipping SafeTensors format (torch not installed)")

                # Try PyTorch binary format - only if PyTorch backend available
                if not model_path and has_torch:
                    # Try various PyTorch model file names
                    pytorch_files = [
                        "model.safetensors",  # Sometimes listed as both
                        "pytorch_model.bin",
                        "model.bin",
                        "model.pt",
                        "model.pth",
                        "model-fp16.safetensors",  # FP16 variants
                        "model-fp16.pt",
                    ]
                    for pytorch_file in pytorch_files:
                        try:
                            _debug(f"Attempting to download PyTorch model ({pytorch_file})...")
                            downloaded_path = hf_hub_download(model_name, pytorch_file)
                            # Check for renamed model first
                            ext = os.path.splitext(pytorch_file)[1]
                            renamed_path = os.path.join(os.path.dirname(downloaded_path), f"{model_short_name}{ext}")

                            # If renamed version already exists, use it
                            if os.path.exists(renamed_path):
                                model_path = renamed_path
                                _debug(f"Found existing renamed model: {renamed_path}")
                            # Otherwise, rename the downloaded file
                            elif downloaded_path != renamed_path:
                                shutil.copy2(downloaded_path, renamed_path)
                                _debug(f"Renamed model to: {renamed_path}")
                                model_path = renamed_path
                            else:
                                model_path = downloaded_path

                            model_format = "pytorch"
                            _debug(f"Using PyTorch model at: {model_path}")
                            break
                        except Exception as e:
                            _debug(f"PyTorch model {pytorch_file} not found: {e}")
                elif not model_path:
                    _debug("Skipping PyTorch format (torch not installed)")

            if not model_path:
                available_backends = []
                if has_onnx:
                    available_backends.append("ONNX Runtime")
                if has_torch:
                    available_backends.append("PyTorch")
                raise RuntimeError(
                    f"No compatible model format found in {model_name}. "
                    f"Available backends: {', '.join(available_backends)}. "
                    f"Tried: ONNX, SafeTensors, PyTorch"
                )

            # Load model based on detected format
            _debug(f"Step 5: Loading model file - path: {model_path}, format: {model_format}")
            _debug(f"About to call _load_wd14_{model_format}...")

            if model_format == "onnx":
                _debug("Calling _load_wd14_onnx...")
                img_size = self._load_wd14_onnx(model_path, model_key)
                _debug(f"_load_wd14_onnx completed, img_size: {img_size}")
            else:  # safetensors or pytorch
                _debug(f"Calling _load_wd14_pytorch with format {model_format}...")
                img_size = self._load_wd14_pytorch(model_path, model_key, model_format)
                _debug(f"_load_wd14_pytorch completed, img_size: {img_size}")

            self.wd14_format = model_format
            self.wd14_img_size = img_size
            self._wd14_loaded = True
            self._current_wd14_model = model_name
            _debug(f"=== WD14 loaded successfully (format: {model_format}, input size: {img_size}) ===")
            return True

        except Exception as e:
            _debug(f"=== WD14 load FAILED ===")
            _debug(f"Error: {e}")
            import traceback
            traceback.print_exc()

            # Clean up any partially loaded state
            self.wd14_model = None
            self.wd14_tags = None
            self.wd14_format = None
            self.wd14_img_size = 448
            self._wd14_loaded = False
            self._current_wd14_model = None

            raise RuntimeError(f"Failed to load WD14: {e}")

    def _load_wd14_onnx(self, model_path: str, model_key: str) -> int:
        """Load WD14 model in ONNX format.

        Returns:
            Input image size expected by the model
        """
        import onnxruntime as ort

        _debug(f"[_load_wd14_onnx] Starting ONNX model load from: {model_path}")
        _debug(f"[_load_wd14_onnx] Model key: {model_key}")

        # Use available execution providers (CUDA if available, otherwise CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        _debug(f"[_load_wd14_onnx] Creating InferenceSession with providers: {providers}")

        try:
            self.wd14_model = ort.InferenceSession(model_path, providers=providers)
            _debug(f"[_load_wd14_onnx] InferenceSession created successfully")
        except Exception as e:
            _debug(f"[_load_wd14_onnx] FAILED to create InferenceSession: {e}")
            raise

        # Get model input details
        _debug(f"[_load_wd14_onnx] Getting input info...")
        input_info = self.wd14_model.get_inputs()[0]
        _debug(f"[_load_wd14_onnx] Model input name: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")

        # Extract input size from shape [batch, channels, height, width]
        img_size = 448  # default
        if input_info.shape and len(input_info.shape) >= 3:
            # Shape is typically ['batch', 3, height, width] or [1, 3, height, width]
            if isinstance(input_info.shape[2], int):
                img_size = input_info.shape[2]
            _debug(f"[_load_wd14_onnx] Detected ONNX input size: {img_size}")

        _debug(f"[_load_wd14_onnx] Completed successfully, returning img_size: {img_size}")
        return img_size

    def _load_wd14_pytorch(self, model_path: str, model_key: str, format_type: str) -> int:
        """Load WD14 model in PyTorch/SafeTensors format.

        Returns:
            Input image size expected by the model
        """
        import torch
        import timm

        _debug(f"[_load_wd14_pytorch] Starting PyTorch model load from: {model_path}")
        _debug(f"[_load_wd14_pytorch] Model key: {model_key}, Format: {format_type}")

        # Determine device
        _debug(f"[_load_wd14_pytorch] Determining device...")
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        _debug(f"[_load_wd14_pytorch] Using device: {device}")

        # Load state dict based on format
        _debug(f"[_load_wd14_pytorch] Loading state_dict from file...")
        try:
            if format_type == "safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                _debug(f"[_load_wd14_pytorch] SafeTensors loaded")
            else:  # pytorch
                state_dict = torch.load(model_path, map_location=device)
                _debug(f"[_load_wd14_pytorch] PyTorch weights loaded")
        except Exception as e:
            _debug(f"[_load_wd14_pytorch] FAILED to load state_dict: {e}")
            raise

        # Infer architecture and image size from state_dict inspection
        _debug(f"[_load_wd14_pytorch] Inferring architecture from state_dict...")
        base_model, img_size = self._infer_wd14_architecture_and_size(state_dict, model_key)
        _debug(f"[_load_wd14_pytorch] Inferred architecture: {base_model}, image size: {img_size}")

        # Create model with correct number of classes and image size
        num_classes = len(self.wd14_tags)
        _debug(f"[_load_wd14_pytorch] Creating timm model with {num_classes} classes, img_size={img_size}...")
        try:
            self.wd14_model = timm.create_model(
                base_model,
                pretrained=False,
                num_classes=num_classes,
                img_size=img_size
            )
            _debug(f"[_load_wd14_pytorch] Model created successfully")
        except Exception as e:
            _debug(f"[_load_wd14_pytorch] FAILED to create model: {e}")
            raise

        # Load weights
        _debug(f"[_load_wd14_pytorch] Loading weights into model...")
        try:
            self.wd14_model.load_state_dict(state_dict, strict=False)
            _debug(f"[_load_wd14_pytorch] Weights loaded")
        except Exception as e:
            _debug(f"[_load_wd14_pytorch] FAILED to load weights: {e}")
            raise

        _debug(f"[_load_wd14_pytorch] Moving model to {device} and setting eval mode...")
        self.wd14_model.to(device).eval()
        _debug(f"[_load_wd14_pytorch] Completed successfully, returning img_size: {img_size}")

        return img_size

    def _infer_wd14_architecture_and_size(self, state_dict: dict, model_key: str) -> Tuple[str, int]:
        """Infer the correct model architecture and image size from state_dict.

        Args:
            state_dict: Model weights dictionary
            model_key: Model key from WD14_MODELS

        Returns:
            Tuple of (timm model name string, image size)
        """
        _debug("Inspecting state_dict to infer architecture and image size...")

        # Try to determine patch size from patch_embed.proj.weight
        patch_size = 16  # default
        if 'patch_embed.proj.weight' in state_dict:
            patch_weight_shape = state_dict['patch_embed.proj.weight'].shape
            _debug(f"patch_embed.proj.weight shape: {patch_weight_shape}")
            # Shape is [out_channels, in_channels, height, width]
            if len(patch_weight_shape) == 4:
                patch_size = patch_weight_shape[2]  # height (should equal width)
                _debug(f"Detected patch size: {patch_size}")

        # Try to determine image size from pos_embed
        img_size = 224  # default
        if 'pos_embed' in state_dict:
            pos_embed_shape = state_dict['pos_embed'].shape
            _debug(f"pos_embed shape: {pos_embed_shape}")
            # Shape is [1, num_patches + 1, embed_dim] where +1 is CLS token
            # OR [1, num_patches, embed_dim] if no CLS token
            num_positions = pos_embed_shape[1]

            # Try with CLS token (num_patches = num_positions - 1)
            num_patches = num_positions - 1
            if patch_size:
                # img_size = sqrt(num_patches) * patch_size
                import math
                patches_per_side = math.sqrt(num_patches)
                if patches_per_side == int(patches_per_side):
                    img_size = int(patches_per_side) * patch_size
                    _debug(f"Detected image size from pos_embed (with CLS): {img_size}")
                else:
                    # Try without CLS token
                    patches_per_side = math.sqrt(num_positions)
                    if patches_per_side == int(patches_per_side):
                        img_size = int(patches_per_side) * patch_size
                        _debug(f"Detected image size from pos_embed (no CLS): {img_size}")

        # Determine base architecture from model key
        # Use generic names WITHOUT size suffix so timm respects the img_size parameter
        base_model = "vit_base_patch16"  # default

        if "vit" in model_key.lower():
            # ViT-based models - use generic names without size suffix
            if patch_size == 14:
                base_model = "vit_base_patch14"
            elif patch_size == 16:
                base_model = "vit_base_patch16"
        elif "swin" in model_key.lower():
            base_model = "swinv2_base_window12to16_192to256"
            img_size = 256  # SwinV2 uses different size
        elif "convnext" in model_key.lower():
            base_model = "convnextv2_base"
            img_size = 224  # ConvNeXt default
        elif "moat" in model_key.lower():
            base_model = "mobilevitv2_150"
            img_size = 256  # MobileVit default

        return base_model, img_size

    def generate_wd14(self, image_path: str, model_key: str = "vit-v2",
                      threshold: float = 0.35) -> str:
        """Generate tags using WD14 tagger.

        Args:
            image_path: Path to image file
            model_key: Key from WD14_MODELS dict
            threshold: Minimum confidence threshold for tags (0.0-1.0)

        Returns:
            Comma-separated tags string
        """
        _debug(f"=== Starting WD14 tagging ===")
        _debug(f"Image path: {image_path}")
        _debug(f"Model key: {model_key}, Threshold: {threshold}")

        # Validate image first
        valid, message = self.validate_image(image_path)
        if not valid:
            raise RuntimeError(f"Image validation failed: {message}")

        # Ensure model is loaded
        self.load_wd14(model_key)

        # Load and preprocess image
        _debug("Opening image with PIL...")
        image = Image.open(image_path).convert("RGB")

        # Resize to model input size (use detected size from model)
        _debug(f"Resizing image to {self.wd14_img_size}x{self.wd14_img_size}")
        image = image.resize((self.wd14_img_size, self.wd14_img_size), Image.Resampling.LANCZOS)

        # Run inference based on model format
        if self.wd14_format == "onnx":
            probs = self._generate_wd14_onnx(image)
        else:  # PyTorch or SafeTensors
            probs = self._generate_wd14_pytorch(image)

        # Get tags above threshold
        # Categories: 0=general, 4=character, 9=rating (we skip rating)
        tags = []
        for i, prob in enumerate(probs):
            if prob >= threshold and i < len(self.wd14_tags):
                tag_info = self.wd14_tags[i]
                # Skip rating tags (category 9)
                if tag_info['category'] != 9:
                    tags.append((tag_info['name'], prob))

        # Sort by probability and format
        tags.sort(key=lambda x: x[1], reverse=True)
        tag_names = [t[0].replace('_', ' ') for t in tags]

        result = ", ".join(tag_names)
        _debug(f"=== Generated {len(tag_names)} tags (format: {self.wd14_format}) ===")
        return result

    def _generate_wd14_onnx(self, image: Image.Image) -> np.ndarray:
        """Run WD14 inference using ONNX Runtime."""
        _debug("Running WD14 ONNX inference...")

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        # Normalize with ImageNet stats (use float32 to match input type)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        # Add batch dimension: HWC -> NHWC (1, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Get model input name
        input_name = self.wd14_model.get_inputs()[0].name

        # Run inference
        outputs = self.wd14_model.run(None, {input_name: img_array})

        # Apply sigmoid to get probabilities
        logits = outputs[0][0]  # Get first batch item
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid function
        return probs

    def _generate_wd14_pytorch(self, image: Image.Image) -> np.ndarray:
        """Run WD14 inference using PyTorch."""
        import torch

        _debug("Running WD14 PyTorch inference...")

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        # HWC -> CHW -> NCHW
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)

        # Get device from model
        device = next(self.wd14_model.parameters()).device
        img_tensor = img_tensor.to(device, dtype=torch.float32)

        # Run inference
        with torch.no_grad():
            outputs = self.wd14_model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        return probs


# Singleton instance for reuse
_generator_instance: Optional[CaptionGenerator] = None


def get_caption_generator() -> CaptionGenerator:
    """Get or create the singleton caption generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = CaptionGenerator()
    return _generator_instance
