"""
Correlate image ratings against embedded ComfyUI generation metadata
(LoRAs + strengths, prompt keywords) to surface what's driving good/bad
results across a rated batch of generations.

Pure logic, no Qt - takes a plain {path: rating} dict so it's testable
without a running app, and reusable outside the thumbnail grid later.
"""

import re
from collections import defaultdict

from comfyui_metadata import extract_comfyui_info

# Common words that add no signal when correlating prompt keywords with rating
STOPWORDS = {
    "the",
    "and",
    "with",
    "for",
    "this",
    "that",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "very",
    "more",
    "most",
    "some",
    "such",
    "than",
    "then",
    "them",
    "they",
    "their",
    "there",
    "these",
    "those",
    "have",
    "has",
    "had",
    "not",
    "but",
    "are",
    "was",
    "were",
    "been",
    "being",
    "you",
    "your",
    "she",
    "her",
    "him",
    "his",
    "its",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "is",
    "as",
    "by",
    "at",
    "or",
}

MIN_KEYWORD_COUNT = 2  # a keyword must show up in at least this many rated images
TOP_N = 10


def analyze_ratings(path_rating_map):
    """
    Args:
        path_rating_map: dict of {image_path: rating (1-5)}

    Returns:
        dict with:
          "num_rated": total images passed in
          "num_with_metadata": how many had embedded ComfyUI metadata
          "overall_avg": average rating across images with metadata
          "lora_stats": list of {"name", "count", "avg_rating",
              "strength_correlation" (float or None), "points"}, sorted by
              how strongly strength predicts rating (most predictive first)
          "keyword_stats_high": top keywords associated with a HIGHER than
              average rating
          "keyword_stats_low": top keywords associated with a LOWER than
              average rating
    """
    num_rated = len(path_rating_map)
    lora_points = defaultdict(list)  # name -> [(strength_or_None, rating), ...]
    keyword_ratings = defaultdict(list)  # keyword -> [rating, ...]
    all_ratings = []
    num_with_metadata = 0

    for path, rating in path_rating_map.items():
        info = extract_comfyui_info(path)
        if not info:
            continue
        num_with_metadata += 1
        all_ratings.append(rating)

        # Note: an image using multiple LoRAs contributes to each LoRA's
        # stats independently - interaction effects between combined LoRAs
        # aren't accounted for. Good enough for a first pass.
        for lora in info["loras"]:
            strength = lora.get("strength_model")
            if strength is None:
                strength = lora.get("strength_clip")
            lora_points[lora["name"]].append((strength, rating))

        prompt_text = info.get("positive_prompt") or ""
        for word in set(_tokenize(prompt_text)):
            keyword_ratings[word].append(rating)

    overall_avg = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    lora_stats = []
    for name, points in lora_points.items():
        ratings_only = [r for _, r in points]
        avg = sum(ratings_only) / len(ratings_only)
        strength_points = [(s, r) for s, r in points if s is not None]
        correlation = _pearson(strength_points) if len(strength_points) >= 3 else None
        lora_stats.append(
            {
                "name": name,
                "count": len(points),
                "avg_rating": avg,
                "strength_correlation": correlation,
                "points": points,
            }
        )
    lora_stats.sort(
        key=lambda d: (
            d["strength_correlation"] is None,
            -abs(d["strength_correlation"])
            if d["strength_correlation"] is not None
            else 0,
        )
    )

    keyword_entries = []
    for word, ratings_list in keyword_ratings.items():
        if len(ratings_list) < MIN_KEYWORD_COUNT:
            continue
        avg = sum(ratings_list) / len(ratings_list)
        keyword_entries.append(
            {
                "keyword": word,
                "count": len(ratings_list),
                "avg_rating": avg,
                "delta": avg - overall_avg,
            }
        )

    keyword_stats_high = sorted(
        (e for e in keyword_entries if e["delta"] > 0),
        key=lambda d: d["delta"],
        reverse=True,
    )[:TOP_N]
    keyword_stats_low = sorted(
        (e for e in keyword_entries if e["delta"] < 0), key=lambda d: d["delta"]
    )[:TOP_N]

    return {
        "num_rated": num_rated,
        "num_with_metadata": num_with_metadata,
        "overall_avg": overall_avg,
        "lora_stats": lora_stats,
        "keyword_stats_high": keyword_stats_high,
        "keyword_stats_low": keyword_stats_low,
    }


def _tokenize(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _pearson(points):
    """Pearson correlation coefficient between two paired value lists."""
    n = len(points)
    if n < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)
