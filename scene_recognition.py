"""
Scene recognition using ResNet-based models trained on Places365.
"""

import cv2
import numpy as np
from pathlib import Path

# Subset of Places365 scene categories (most useful for photo organization)
SCENE_CATEGORIES = {
    # Outdoor nature
    "beach", "coast", "lake", "ocean", "river", "waterfall",
    "forest", "mountain", "valley", "canyon", "desert", "field",
    "sky", "sunset", "sunrise",

    # Urban outdoor
    "street", "alley", "highway", "bridge", "parking_lot",
    "plaza", "park", "garden", "yard", "patio",

    # Buildings
    "building", "skyscraper", "tower", "castle", "house",
    "apartment_building", "barn", "church", "temple", "mosque",

    # Indoor residential
    "bedroom", "living_room", "dining_room", "kitchen",
    "bathroom", "closet", "garage", "basement", "attic",

    # Indoor commercial
    "restaurant", "cafe", "bar", "store", "shop",
    "office", "lobby", "corridor", "elevator",

    # Entertainment/Leisure
    "gym", "stadium", "theater", "museum", "gallery",
    "playground", "poolroom", "bowling_alley",

    # Transportation
    "airport", "train_station", "subway", "bus_station",

    # Special
    "underwater", "snowfield", "ice", "rainforest"
}

# Categorize scenes for easier filtering
SCENE_GROUPS = {
    "outdoor_nature": ["beach", "coast", "lake", "ocean", "river", "waterfall",
                       "forest", "mountain", "valley", "canyon", "desert", "field",
                       "sky", "sunset", "sunrise"],
    "outdoor_urban": ["street", "alley", "highway", "bridge", "parking_lot",
                      "plaza", "park", "garden", "yard", "patio"],
    "indoor_home": ["bedroom", "living_room", "dining_room", "kitchen",
                    "bathroom", "closet", "garage", "basement", "attic"],
    "indoor_commercial": ["restaurant", "cafe", "bar", "store", "shop",
                          "office", "lobby", "corridor", "elevator"],
    "entertainment": ["gym", "stadium", "theater", "museum", "gallery",
                     "playground", "poolroom", "bowling_alley"],
    "transportation": ["airport", "train_station", "subway", "bus_station"],
}


class SceneRecognizer:
    """Recognizes scenes in images using ResNet-based classifier."""

    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.scene_names = []
        self._load_model()
        self._cache = {}

    def _load_model(self):
        """Load Places365 ResNet model (CPU-friendly)."""
        models_dir = Path(__file__).parent / "models"

        # Paths to model files (ONNX format required for OpenCV)
        model_path = models_dir / "resnet18_places365.onnx"
        labels_path = models_dir / "IO_places365.txt"

        # Check for PyTorch model as fallback indicator
        pth_model = models_dir / "resnet18_places365.pth.tar"

        if not model_path.exists():
            if pth_model.exists():
                print(f"\n{'='*60}")
                print("NOTE: Places365 PyTorch model found but ONNX needed")
                print(f"Location: {models_dir}")
                print("\nScene recognition requires ONNX format for OpenCV.")
                print("The PyTorch model would need conversion (requires torch/torchvision).")
                print("\nFor now, scene recognition filtering is disabled.")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print("NOTE: Places365 model not found")
                print(f"Location: {models_dir}")
                print("\nTo enable scene recognition filtering, download the ONNX model.")
                print(f"{'='*60}\n")
            self.net = None
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Load scene names from IO_places365.txt format
            # Format: /indoor/bedroom 0 ... (we just need the first part)
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    self.scene_names = []
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            # Extract scene name from format: /category/scene_name
                            scene = line.split()[0].strip('/').split('/')[-1]
                            self.scene_names.append(scene)
            else:
                # Fallback to subset
                self.scene_names = list(SCENE_CATEGORIES)

            print("✓ Places365 scene recognizer loaded successfully (CPU mode)")
        except Exception as e:
            print(f"✗ Failed to load Places365 model: {e}")
            self.net = None

    def recognize_scene(self, image_path, top_k=5):
        """
        Recognize scenes in an image.

        Returns:
            list of tuples: [(scene_name, confidence), ...] for top K predictions
        """
        if self.net is None:
            return []

        # Check cache
        cache_key = str(image_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                return []

            # Resize to 224x224 (standard for ResNet)
            blob = cv2.dnn.blobFromImage(
                img,
                scalefactor=1.0/255.0,
                size=(224, 224),
                mean=(0.485, 0.456, 0.406),
                swapRB=True,
                crop=False
            )

            # Set input and forward pass
            self.net.setInput(blob)
            predictions = self.net.forward()

            # Get top K predictions
            predictions = predictions.flatten()
            top_indices = np.argsort(predictions)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(self.scene_names):
                    scene_name = self.scene_names[idx]
                    confidence = float(predictions[idx])

                    if confidence >= self.confidence_threshold:
                        results.append((scene_name, confidence))

            # Cache results
            self._cache[cache_key] = results
            return results

        except Exception as e:
            print(f"Error recognizing scene in {image_path}: {e}")
            return []

    def has_scene(self, image_path, scene_name):
        """Check if image is classified as a specific scene."""
        results = self.recognize_scene(image_path)
        return any(scene[0] == scene_name for scene in results)

    def has_scene_group(self, image_path, group):
        """Check if image belongs to a scene group."""
        if group not in SCENE_GROUPS:
            return False

        results = self.recognize_scene(image_path)
        detected_scenes = [scene[0] for scene in results]

        return any(scene in SCENE_GROUPS[group] for scene in detected_scenes)

    def is_outdoor(self, image_path):
        """Check if image is an outdoor scene."""
        return (self.has_scene_group(image_path, "outdoor_nature") or
                self.has_scene_group(image_path, "outdoor_urban"))

    def is_indoor(self, image_path):
        """Check if image is an indoor scene."""
        return (self.has_scene_group(image_path, "indoor_home") or
                self.has_scene_group(image_path, "indoor_commercial") or
                self.has_scene_group(image_path, "entertainment"))

    def clear_cache(self):
        """Clear recognition cache."""
        self._cache.clear()

    def set_confidence(self, threshold):
        """Update confidence threshold and clear cache."""
        self.confidence_threshold = threshold
        self.clear_cache()


# Singleton instance
_recognizer = None

def get_recognizer(confidence_threshold=0.3):
    """Get or create the scene recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = SceneRecognizer(confidence_threshold)
    else:
        _recognizer.set_confidence(confidence_threshold)
    return _recognizer
