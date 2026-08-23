"""
Composition similarity descriptors, for ordering images by structural
diversity independent of colour.
"""

import cv2
import numpy as np


class CompositionAnalyzer:
    """Computes and caches colour-independent composition descriptors.

    Each image is reduced to a HOG (Histogram of Oriented Gradients)
    fingerprint over a fixed-size grayscale version of the frame. HOG
    captures edge layout/orientation - framing, pose, cropping - while
    discarding colour and fine texture, which is what "composition"
    means for this feature.
    """

    DESCRIPTOR_SIZE = 128  # square resize before HOG; ignores aspect ratio on purpose
    # so descriptors from differently-cropped candidates remain comparable
    _WIN_SIZE = (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
    _BLOCK_SIZE = (16, 16)
    _BLOCK_STRIDE = (8, 8)
    _CELL_SIZE = (8, 8)
    _N_BINS = 9

    def __init__(self):
        self._cache = {}
        self._hog = cv2.HOGDescriptor(
            self._WIN_SIZE,
            self._BLOCK_SIZE,
            self._BLOCK_STRIDE,
            self._CELL_SIZE,
            self._N_BINS,
        )

    def get_descriptor(self, image_path):
        """
        Get a colour-independent composition descriptor for an image.

        Args:
            image_path: Path to the image file

        Returns:
            numpy.ndarray or None: 1-D HOG descriptor vector, or None if the
                image could not be read.
        """
        if image_path in self._cache:
            return self._cache[image_path]

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._cache[image_path] = None
                return None

            resized = cv2.resize(
                img,
                (self.DESCRIPTOR_SIZE, self.DESCRIPTOR_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            # Equalize so exposure/lighting differences don't dominate the
            # structural signal we actually care about.
            equalized = cv2.equalizeHist(resized)

            descriptor = self._hog.compute(equalized)
            if descriptor is None:
                self._cache[image_path] = None
                return None

            descriptor = descriptor.flatten().astype(np.float32)
            self._cache[image_path] = descriptor
            return descriptor

        except Exception as e:
            print(f"Error computing composition descriptor for {image_path}: {e}")
            self._cache[image_path] = None
            return None

    def clear_cache(self):
        """Clear the descriptor cache."""
        self._cache.clear()


def order_by_diversity(paths, descriptors):
    """
    Order items by greedy farthest-point sampling over their descriptors.

    Starts from the item closest to the centroid (the most "typical" one,
    for a deterministic start) and repeatedly appends whichever remaining
    item is most dissimilar to everything already placed. Taking the first
    N items of the result gives a maximally-spread subset of size N -
    useful for picking a diverse training set instead of near-duplicate
    framings.

    Args:
        paths: list of identifiers (e.g. file paths), parallel to descriptors
        descriptors: list of 1-D numpy arrays, one per path, same order

    Returns:
        list: `paths` reordered most-typical-seed-then-most-diverse-first
    """
    n = len(paths)
    if n <= 2:
        return list(paths)

    vectors = np.stack(descriptors).astype(np.float32)

    centroid = vectors.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(vectors - centroid, axis=1)))

    selected = [seed]
    remaining = set(range(n)) - {seed}
    min_dist = np.linalg.norm(vectors - vectors[seed], axis=1)

    while remaining:
        next_idx = max(remaining, key=lambda i: min_dist[i])
        selected.append(next_idx)
        remaining.discard(next_idx)
        if remaining:
            new_dist = np.linalg.norm(vectors - vectors[next_idx], axis=1)
            min_dist = np.minimum(min_dist, new_dist)

    return [paths[i] for i in selected]
