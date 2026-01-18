"""
Object detection using YOLO models for filtering images by content.
"""

import os
import cv2
import numpy as np
from pathlib import Path

# COCO dataset class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Categorize objects for easier filtering
OBJECT_CATEGORIES = {
    "people": ["person"],
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"],
}


class ObjectDetector:
    """Detects objects in images using YOLO."""

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.output_layers = None
        self._yolo_model = None  # For ultralytics models
        self._is_ultralytics = False
        self._load_model()
        self._cache = {}

    def _load_model(self):
        """Load YOLOv4-tiny model (CPU-friendly)."""
        models_dir = Path(__file__).parent / "models"

        # Check for custom model first (set via set_custom_model)
        if hasattr(self, '_custom_model_path') and self._custom_model_path:
            try:
                self._load_custom_model(self._custom_model_path)
                return
            except Exception as e:
                print(f"Failed to load custom model: {e}")
                print("Falling back to default YOLO model...")

        # Paths to default model files
        weights_path = models_dir / "yolov4-tiny.weights"
        config_path = models_dir / "yolov4-tiny.cfg"

        if not weights_path.exists() or not config_path.exists():
            print(f"\n{'='*60}")
            print("WARNING: YOLO model files not found!")
            print(f"Location: {models_dir}")
            print("\nTo enable object detection filtering, please run:")
            print("  cd models")
            print("  python3 download_models.py")
            print(f"{'='*60}\n")
            self.net = None
            self.model_loaded = False
            return

        try:
            self.net = cv2.dnn.readNet(str(weights_path), str(config_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            self.model_loaded = True
            print("✓ YOLO object detector loaded successfully (CPU mode)")
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            self.net = None
            self.model_loaded = False

    def _load_custom_model(self, model_path):
        """Load a custom YOLO model (.pt, .pth, .safetensors, .weights, or .onnx)."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        suffix = model_path.suffix.lower()

        if suffix in ['.weights']:
            # Darknet weights format - need .cfg file
            cfg_path = model_path.with_suffix('.cfg')
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file not found: {cfg_path}")

            self.net = cv2.dnn.readNet(str(model_path), str(cfg_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.model_loaded = True
            print(f"✓ Custom YOLO model loaded: {model_path.name}")

        elif suffix == '.onnx':
            # ONNX format
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # For ONNX, output layers might be different
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            self.model_loaded = True
            print(f"✓ Custom YOLO model loaded (ONNX): {model_path.name}")

        elif suffix in ['.pt', '.pth', '.safetensors']:
            # PyTorch formats - requires ultralytics/torch
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(str(model_path))
                self.model_loaded = True
                self._is_ultralytics = True
                print(f"✓ Custom YOLO model loaded (Ultralytics): {model_path.name}")
            except ImportError:
                raise ImportError(
                    "PyTorch YOLO models require 'ultralytics' package.\n"
                    "Install with: pip install ultralytics"
                )
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def set_custom_model(self, model_path):
        """Set and load a custom YOLO model."""
        self._custom_model_path = model_path
        self._load_model()

    def detect_objects(self, image_path):
        """
        Detect objects in an image.

        Returns:
            list of tuples: [(class_name, confidence), ...]
        """
        if self.net is None and self._yolo_model is None:
            return []

        # Check cache
        cache_key = str(image_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Ultralytics YOLO models
            if self._is_ultralytics and self._yolo_model is not None:
                results = self._yolo_model.predict(
                    str(image_path),
                    conf=self.confidence_threshold,
                    verbose=False
                )

                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            conf = float(boxes.conf[i])
                            if conf > self.confidence_threshold:
                                # Get class name if available
                                class_id = int(boxes.cls[i])
                                if hasattr(result, 'names') and class_id in result.names:
                                    class_name = result.names[class_id]
                                else:
                                    class_name = f"object_{class_id}"
                                detections.append((class_name, conf))

                # Cache results
                self._cache[cache_key] = detections
                return detections

            # OpenCV DNN models (Darknet/ONNX)
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return []

            height, width = img.shape[:2]

            # Create blob (416x416 for YOLOv4-tiny)
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)

            # Forward pass
            outputs = self.net.forward(self.output_layers)

            # Parse detections
            detections = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        class_name = COCO_CLASSES[class_id]
                        detections.append((class_name, float(confidence)))

            # Cache results
            self._cache[cache_key] = detections
            return detections

        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return []

    def has_object(self, image_path, object_name):
        """Check if image contains a specific object."""
        detections = self.detect_objects(image_path)
        return any(obj[0] == object_name for obj in detections)

    def has_category(self, image_path, category):
        """Check if image contains any object from a category."""
        if category not in OBJECT_CATEGORIES:
            return False

        detections = self.detect_objects(image_path)
        detected_objects = [obj[0] for obj in detections]

        return any(obj in OBJECT_CATEGORIES[category] for obj in detected_objects)

    def get_detected_categories(self, image_path):
        """Get all categories detected in the image."""
        detections = self.detect_objects(image_path)
        detected_objects = [obj[0] for obj in detections]

        categories = set()
        for category, objects in OBJECT_CATEGORIES.items():
            if any(obj in objects for obj in detected_objects):
                categories.add(category)

        return list(categories)

    def clear_cache(self):
        """Clear detection cache."""
        self._cache.clear()

    def set_confidence(self, threshold):
        """Update confidence threshold and clear cache."""
        self.confidence_threshold = threshold
        self.clear_cache()


# Singleton instance
_detector = None

def get_detector(confidence_threshold=0.5):
    """Get or create the object detector instance."""
    global _detector
    if _detector is None:
        _detector = ObjectDetector(confidence_threshold)
    else:
        _detector.set_confidence(confidence_threshold)
    return _detector
