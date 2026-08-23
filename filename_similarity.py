"""
Shared filename-similarity matching helpers.

Used to pair up images across two folders when filenames aren't identical
but are related (e.g. an original and a derived/cropped/renamed copy).
"""

import os

VALID_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".avif",
    ".tiff",
    ".bmp",
    ".tn3",
}


def scan_sorted(directory, sort_mode="name"):
    """Return a sorted list of full image paths in *directory*."""
    try:
        files = [
            f
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
    except OSError:
        return []
    if sort_mode == "date":
        files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    else:
        files.sort(key=str.lower)
    return [os.path.join(directory, f) for f in files]


def similarity(a, b):
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 3:
        return 0.0
    ratio = len(shorter) / len(longer)
    if longer.startswith(shorter) and ratio >= 0.3:
        return 0.95
    if longer.endswith(shorter) and ratio >= 0.3:
        return 0.90
    if shorter in longer and ratio >= 0.4:
        return 0.80
    return 0.0


def pair_files(files_a, files_b, min_score=0.5):
    """Return [(path_a, path_b, score), ...] matched by filename similarity, in A-order."""
    bases_b = [(os.path.splitext(os.path.basename(p))[0], p) for p in files_b]
    used = set()
    pairs = []
    for pa in files_a:
        base_a = os.path.splitext(os.path.basename(pa))[0]
        best_score, best_idx = 0.0, None
        for j, (base_b, _) in enumerate(bases_b):
            if j in used:
                continue
            s = similarity(base_a, base_b)
            if s > best_score:
                best_score, best_idx = s, j
        if best_idx is not None and best_score >= min_score:
            pairs.append((pa, bases_b[best_idx][1], best_score))
            used.add(best_idx)
    return pairs


def auto_pair_positions(files_a, files_b, min_score=0.5):
    """
    Return (idx_a, idx_b) — 0-based positions of the best-matching first pair
    found by filename similarity, for use in position mode Auto-align.
    """
    bases_b = [os.path.splitext(os.path.basename(p))[0] for p in files_b]
    used_b = set()
    for i, pa in enumerate(files_a):
        base_a = os.path.splitext(os.path.basename(pa))[0]
        best_score, best_j = 0.0, None
        for j, base_b in enumerate(bases_b):
            if j in used_b:
                continue
            s = similarity(base_a, base_b)
            if s > best_score:
                best_score, best_j = s, j
        if best_j is not None and best_score >= min_score:
            return i, best_j
        if best_j is not None:
            used_b.add(best_j)
    return 0, 0
