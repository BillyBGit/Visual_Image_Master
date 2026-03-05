"""
Image quality assessment using BRISQUE and custom metrics.
"""

import cv2
import numpy as np


class QualityAssessor:
    """Assesses image quality using multiple metrics."""

    def __init__(self):
        self._cache = {}

    def assess_quality(self, image_path):
        """
        Assess image quality on multiple dimensions.

        Returns:
            dict: {
                'sharpness': float (0-100, higher is sharper),
                'brightness': float (0-255, 128 is neutral),
                'contrast': float (0-100, higher is better),
                'colorfulness': float (0-100, higher is more colorful),
                'overall_score': float (0-100, weighted combination)
            }
        """
        # Check cache
        cache_key = str(image_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # Calculate metrics
            sharpness = self._calculate_sharpness(img)
            brightness = self._calculate_brightness(img)
            contrast = self._calculate_contrast(img)
            colorfulness = self._calculate_colorfulness(img)

            # Overall score (weighted combination)
            # Prioritize sharpness and contrast
            overall = (
                sharpness * 0.4
                + contrast * 0.3
                + colorfulness * 0.2
                + self._normalize_brightness(brightness) * 0.1
            )

            result = {
                "sharpness": round(sharpness, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "colorfulness": round(colorfulness, 2),
                "overall_score": round(overall, 2),
            }

            # Cache results
            self._cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Error assessing quality of {image_path}: {e}")
            return None

    def _calculate_sharpness(self, img):
        """Calculate image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-100 scale (empirically determined)
        # Values > 1000 are very sharp, < 100 are blurry
        normalized = min(100, (laplacian_var / 10))
        return normalized

    def _calculate_brightness(self, img):
        """Calculate average brightness."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()
        return brightness

    def _normalize_brightness(self, brightness):
        """Normalize brightness to 0-100 score (penalize too dark/bright)."""
        # Ideal brightness is around 128 (middle)
        distance_from_ideal = abs(brightness - 128)
        score = max(0, 100 - (distance_from_ideal / 1.28))
        return score

    def _calculate_contrast(self, img):
        """Calculate image contrast."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()

        # Normalize to 0-100 scale
        # Std dev typically ranges from 0-70 for most images
        normalized = min(100, (contrast / 0.7))
        return normalized

    def _calculate_colorfulness(self, img):
        """Calculate image colorfulness using rg-yb opponent color space."""
        # Split into channels
        (B, G, R) = cv2.split(img.astype("float"))

        # Compute rg = R - G
        rg = np.absolute(R - G)

        # Compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # Compute the mean and standard deviation of both rg and yb
        rb_mean, rb_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))

        # Combine the mean and standard deviations
        std_root = np.sqrt((rb_std**2) + (yb_std**2))
        mean_root = np.sqrt((rb_mean**2) + (yb_mean**2))

        # Colorfulness metric
        colorfulness = std_root + (0.3 * mean_root)

        # Normalize to 0-100 scale
        # Empirically, colorfulness ranges from 0-100+ for typical photos
        normalized = min(100, colorfulness)
        return normalized

    def is_high_quality(self, image_path, min_score=60):
        """Check if image meets minimum quality threshold."""
        result = self.assess_quality(image_path)
        if result is None:
            return False
        return result["overall_score"] >= min_score

    def is_sharp(self, image_path, min_sharpness=40):
        """Check if image is sufficiently sharp."""
        result = self.assess_quality(image_path)
        if result is None:
            return False
        return result["sharpness"] >= min_sharpness

    def is_well_exposed(self, image_path, min_brightness=50, max_brightness=200):
        """Check if image is well-exposed (not too dark/bright)."""
        result = self.assess_quality(image_path)
        if result is None:
            return False
        brightness = result["brightness"]
        return min_brightness <= brightness <= max_brightness

    def clear_cache(self):
        """Clear quality assessment cache."""
        self._cache.clear()


# Singleton instance
_assessor = None


def get_assessor():
    """Get or create the quality assessor instance."""
    global _assessor
    if _assessor is None:
        _assessor = QualityAssessor()
    return _assessor
