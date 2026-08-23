"""
Content-based image matching — pairs a crop/extract to the source image it
was taken from, even when filenames carry no relation to each other.

Pipeline (shared with find_unmatched.py, which uses it for CLI reporting):
  1. SIFT (or ORB fallback) keypoints extracted from each image.
  2. Lowe ratio test filters weak descriptor matches.
  3. RANSAC homography fit: only keypoints that agree on a geometrically
     consistent crop-to-source transform count as inliers. Coincidental
     matches across unrelated images scatter randomly and are rejected.
  4. A crop is "matched" to whichever source has the highest inlier count,
     provided it clears --min-matches.

Typical true matches produce 20-200+ inliers; false positives rarely exceed
5-8, so the default min_matches=8 is a reasonable starting point.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".avif"}


def image_files(directory):
    directory = Path(directory)
    return sorted(
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def load_gray(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def make_detector(nfeatures=2000):
    """Return (detector, matcher, name) — SIFT if available, else ORB."""
    if hasattr(cv2, "SIFT_create"):
        return (
            cv2.SIFT_create(nfeatures=nfeatures),
            cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
            "SIFT",
        )
    return (
        cv2.ORB_create(nfeatures=nfeatures),
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
        "ORB",
    )


def _kp_to_array(kps):
    if not kps:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave] for k in kps],
        dtype=np.float32,
    )


def _array_to_kp(arr):
    return [
        cv2.KeyPoint(
            x=float(r[0]),
            y=float(r[1]),
            size=float(r[2]),
            angle=float(r[3]),
            response=float(r[4]),
            octave=int(r[5]),
        )
        for r in arr
    ]


def get_features(path, detector, cache_dir=None):
    """Return (keypoints, descriptors), using a .npz disk cache when cache_dir is set."""
    path = Path(path)
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        mtime = path.stat().st_mtime
        key = hashlib.md5(f"{path}:{mtime}".encode()).hexdigest()
        cache_file = cache_dir / f"{key}.npz"
        if cache_file.exists():
            data = np.load(str(cache_file), allow_pickle=True)
            kps = _array_to_kp(data["kp"])
            des = data["des"] if data["des"].shape != () else None
            return kps, des

    gray = load_gray(path)
    if gray is None:
        kps, des = [], None
    else:
        kps, des = detector.detectAndCompute(gray, None)
        if kps is None:
            kps = []

    if cache_file is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(cache_file),
            kp=_kp_to_array(kps),
            des=des if des is not None else np.array(None, dtype=object),
        )

    return kps, des


def inlier_count(kp_a, des_a, kp_b, des_b, matcher, ratio=0.75, ransac_thresh=5.0):
    """Return the number of RANSAC inliers between image A and image B."""
    if des_a is None or des_b is None:
        return 0
    if len(des_a) < 2 or len(des_b) < 2:
        return 0

    try:
        pairs = matcher.knnMatch(des_a, des_b, k=2)
    except cv2.error:
        return 0

    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if len(good) < 4:
        return 0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0

    return int(mask.sum())


def compute_hist(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def hist_similarity(ha, hb):
    if ha is None or hb is None:
        return 0.0
    denom = np.linalg.norm(ha) * np.linalg.norm(hb)
    return float(np.dot(ha, hb) / (denom + 1e-9))


def match_extracts_to_originals(
    originals,
    extracts,
    min_matches=8,
    ratio=0.75,
    ransac_thresh=5.0,
    nfeatures=2000,
    hist_prefilter=0.0,
    cache_dir=None,
    progress_callback=None,
    should_cancel=None,
):
    """
    For each extract, find the best-matching original by SIFT/RANSAC content
    matching. Returns [(original_path, extract_path, inlier_count), ...] for
    extracts that cleared min_matches, in extract order.

    progress_callback(done, total, message) is called periodically if given.
    should_cancel() may be polled to abort early, returning results so far.
    """
    detector, matcher, _name = make_detector(nfeatures)
    originals = [str(p) for p in originals]
    extracts = [str(p) for p in extracts]
    total = len(originals) + len(extracts)
    done = 0

    orig_data = []
    for path in originals:
        if should_cancel and should_cancel():
            return []
        kps, des = get_features(path, detector, cache_dir)
        orig_data.append((path, kps, des))
        done += 1
        if progress_callback:
            progress_callback(done, total, f"Indexing originals: {Path(path).name}")

    orig_hists = None
    if hist_prefilter > 0:
        orig_hists = [compute_hist(p) for p, _, _ in orig_data]

    results = []
    for extract_path in extracts:
        if should_cancel and should_cancel():
            return results
        kp_e, des_e = get_features(extract_path, detector, cache_dir)

        best_score, best_orig = 0, None
        if des_e is not None:
            e_hist = compute_hist(extract_path) if hist_prefilter > 0 else None
            for k, (orig_path, kp_o, des_o) in enumerate(orig_data):
                if hist_prefilter > 0:
                    sim = hist_similarity(e_hist, orig_hists[k])
                    if sim < hist_prefilter:
                        continue
                n = inlier_count(
                    kp_e, des_e, kp_o, des_o, matcher, ratio, ransac_thresh
                )
                if n > best_score:
                    best_score, best_orig = n, orig_path

        if best_orig is not None and best_score >= min_matches:
            results.append((best_orig, extract_path, best_score))

        done += 1
        if progress_callback:
            progress_callback(
                done, total, f"Matching extracts: {Path(extract_path).name}"
            )

    return results
