"""
Face detection and image quality utilities using OpenCV.
Supports both Haar Cascade and DNN-based detection methods.
"""

import cv2
import os

# Face detection presets
FACE_DETECTION_PRESETS = {
    "fast": {
        "method": "haar",
        "scale_factor": 1.2,
        "min_neighbors": 4,
        "min_size": 40,
        "description": "Fast Haar Cascade detection (less accurate)"
    },
    "balanced": {
        "method": "dnn",
        "confidence": 0.5,
        "min_size": 30,
        "description": "DNN-based detection (good balance)"
    },
    "thorough": {
        "method": "both",
        "haar_scale_factor": 1.1,
        "haar_min_neighbors": 3,
        "dnn_confidence": 0.4,
        "min_size": 25,
        "description": "Use both detectors (most accurate)"
    }
}


class FaceDetector:
    """Face detector supporting multiple detection methods (Haar Cascade and DNN)."""

    def __init__(self, preset="balanced"):
        """
        Initialize face detector with specified preset.

        Args:
            preset: One of "fast", "balanced", or "thorough"
        """
        # Load Haar Cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Try to load DNN model
        self.dnn_net = None
        self._load_dnn_model()

        # Set detection preset
        self.preset = preset
        self._apply_preset(preset)

        # Cache for face detection results
        self._cache = {}

    def _load_dnn_model(self):
        """Load the DNN face detection model if available."""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, "models")

            prototxt_path = os.path.join(models_dir, "deploy.prototxt")
            model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("✓ DNN face detection model loaded successfully")
            else:
                print("⚠ DNN model files not found. Run models/download_models.py to download them.")
                print("  Falling back to Haar Cascade detection only.")
        except Exception as e:
            print(f"⚠ Could not load DNN model: {e}")
            print("  Falling back to Haar Cascade detection only.")
            self.dnn_net = None

    def _apply_preset(self, preset):
        """Apply detection parameters from preset."""
        if preset not in FACE_DETECTION_PRESETS:
            preset = "balanced"

        self.preset = preset
        preset_config = FACE_DETECTION_PRESETS[preset]

        # Store configuration
        self.method = preset_config["method"]

        # Haar Cascade parameters
        self.haar_scale_factor = preset_config.get("scale_factor", preset_config.get("haar_scale_factor", 1.1))
        self.haar_min_neighbors = preset_config.get("min_neighbors", preset_config.get("haar_min_neighbors", 5))

        # DNN parameters
        self.dnn_confidence = preset_config.get("confidence", preset_config.get("dnn_confidence", 0.5))

        # Common parameters
        self.min_size = preset_config.get("min_size", 30)

    def set_preset(self, preset):
        """
        Change detection preset.

        Args:
            preset: One of "fast", "balanced", or "thorough"
        """
        self._apply_preset(preset)
        self.clear_cache()  # Clear cache when settings change

    def get_preset_info(self):
        """Get information about current preset."""
        if self.preset in FACE_DETECTION_PRESETS:
            return FACE_DETECTION_PRESETS[self.preset]
        return None

    def detect_faces(self, image_path):
        """
        Detect faces in an image and return their locations.

        Args:
            image_path: Path to the image file

        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        # Check cache first (using different key to separate from has_face cache)
        cache_key = f"{image_path}_{self.preset}_faces"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                self._cache[cache_key] = []
                return []

            faces = []

            # Choose detection method based on preset
            if self.method == "haar" or self.method == "both":
                faces = self._detect_haar_faces(img)

            if len(faces) == 0 and (self.method == "dnn" or self.method == "both"):
                if self.dnn_net is not None:
                    faces = self._detect_dnn_faces(img)
                elif self.method == "dnn":
                    # DNN requested but not available, fall back to Haar
                    faces = self._detect_haar_faces(img)

            # Cache and return result
            self._cache[cache_key] = faces
            return faces

        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            self._cache[cache_key] = []
            return []

    def has_face(self, image_path):
        """
        Check if an image contains at least one face.

        Args:
            image_path: Path to the image file

        Returns:
            bool: True if face(s) detected, False otherwise
        """
        # Check cache first
        cache_key = f"{image_path}_{self.preset}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                self._cache[cache_key] = False
                return False

            has_faces = False

            # Choose detection method based on preset
            if self.method == "haar" or self.method == "both":
                has_faces = self._detect_haar(img)

            if not has_faces and (self.method == "dnn" or self.method == "both"):
                if self.dnn_net is not None:
                    has_faces = self._detect_dnn(img)
                elif self.method == "dnn":
                    # DNN requested but not available, fall back to Haar
                    has_faces = self._detect_haar(img)

            # Cache and return result
            self._cache[cache_key] = has_faces
            return has_faces

        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            self._cache[cache_key] = False
            return False

    def _detect_haar_faces(self, img):
        """Detect faces using Haar Cascade classifier and return face locations."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.haar_scale_factor,
            minNeighbors=self.haar_min_neighbors,
            minSize=(self.min_size, self.min_size)
        )

        # Convert to list of tuples
        return [tuple(face) for face in faces]

    def _detect_dnn_faces(self, img):
        """Detect faces using DNN-based detector and return face locations."""
        if self.dnn_net is None:
            return []

        h, w = img.shape[:2]

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Pass the blob through the network
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        # Collect all detections that meet the confidence threshold
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.dnn_confidence:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")

                # Check minimum size
                width = x2 - x1
                height = y2 - y1
                if width >= self.min_size and height >= self.min_size:
                    # Store as (x, y, w, h) to match Haar format
                    faces.append((x1, y1, width, height))

        return faces

    def _detect_haar(self, img):
        """Detect faces using Haar Cascade classifier (returns boolean)."""
        faces = self._detect_haar_faces(img)
        return len(faces) > 0

    def _detect_dnn(self, img):
        """Detect faces using DNN-based detector (returns boolean)."""
        faces = self._detect_dnn_faces(img)
        return len(faces) > 0

    def clear_cache(self):
        """Clear the face detection cache."""
        self._cache.clear()


class ImageQualityAnalyzer:
    """Analyzes image quality metrics like blur/sharpness."""

    def __init__(self):
        # Cache for blur scores
        self._blur_cache = {}

    def get_blur_score(self, image_path):
        """
        Calculate blur/sharpness score using Laplacian variance.
        Higher values = sharper image, lower values = more blurry.

        Args:
            image_path: Path to the image file

        Returns:
            float: Blur score (higher is sharper)
        """
        # Check cache first
        if image_path in self._blur_cache:
            return self._blur_cache[image_path]

        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                self._blur_cache[image_path] = 0.0
                return 0.0

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate Laplacian variance (measure of sharpness)
            # Higher variance = sharper image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Cache and return result
            self._blur_cache[image_path] = laplacian_var
            return laplacian_var

        except Exception as e:
            print(f"Error analyzing blur in {image_path}: {e}")
            self._blur_cache[image_path] = 0.0
            return 0.0

    def clear_cache(self):
        """Clear the blur score cache."""
        self._blur_cache.clear()
