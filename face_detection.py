"""
Face detection and image quality utilities using OpenCV.
"""

import cv2
import os


class FaceDetector:
    """Face detector using OpenCV's Haar Cascade classifier."""

    def __init__(self):
        # Load the pre-trained Haar Cascade classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Cache for face detection results
        self._cache = {}

    def has_face(self, image_path):
        """
        Check if an image contains at least one face.

        Args:
            image_path: Path to the image file

        Returns:
            bool: True if face(s) detected, False otherwise
        """
        # Check cache first
        if image_path in self._cache:
            return self._cache[image_path]

        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                self._cache[image_path] = False
                return False

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Cache and return result
            has_faces = len(faces) > 0
            self._cache[image_path] = has_faces
            return has_faces

        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            self._cache[image_path] = False
            return False

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
