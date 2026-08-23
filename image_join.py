"""
Image join engine for Visual Image Master 2026.

Stitches a rectangular grid of images (e.g. portrait crops split from one
larger landscape photo) back into a single combined image.

No Qt dependency - pure OpenCV and numpy.
"""

import cv2
import numpy as np


def _resize_to_height(img_bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h == target_h:
        return img_bgr
    new_w = max(1, round(w * (target_h / h)))
    return cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _resize_to_width(img_bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == target_w:
        return img_bgr
    new_h = max(1, round(h * (target_w / w)))
    return cv2.resize(img_bgr, (target_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def stitch_grid(grid: list[list[np.ndarray]]) -> np.ndarray:
    """
    Combine a rectangular grid of BGR images into one image.

    grid — a list of rows, each row a list of BGR numpy arrays in left-to-right
    order; every row must have the same number of columns. Images within a row
    are resized to a common height (the row's max) before being concatenated
    horizontally; the resulting row strips are then resized to a common width
    (the grid's max) before being concatenated vertically. This tolerates
    small crop-boundary mismatches between pieces without failing.
    """
    row_strips = []
    for row in grid:
        target_h = max(img.shape[0] for img in row)
        resized = [_resize_to_height(img, target_h) for img in row]
        row_strips.append(cv2.hconcat(resized))

    target_w = max(strip.shape[1] for strip in row_strips)
    resized_strips = [_resize_to_width(strip, target_w) for strip in row_strips]
    return cv2.vconcat(resized_strips)
