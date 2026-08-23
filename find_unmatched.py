#!/usr/bin/env python3
"""
find_unmatched.py — Find source images with no corresponding crop.

Given a directory of crops (images that are rescaled portions of source images)
and a directory of source images, report which source images have no matching
crop.

Matching pipeline:
  1. SIFT keypoints extracted from each image.
  2. Lowe ratio test filters weak descriptor matches.
  3. RANSAC homography fit: only keypoints that agree on a geometrically
     consistent crop-to-source transform count as inliers.  Coincidental
     matches across unrelated images scatter randomly and are rejected.
  4. Source is "matched" when inlier count >= --min-matches.

Use --verbose to see inlier counts per pair and tune --min-matches for your
dataset.  Typical true matches produce 20–200+ inliers; false positives rarely
exceed 5–8.

Usage:
    python find_unmatched.py CROPS_DIR SOURCES_DIR [options]

Exit code:
    0  — all sources are matched
    1  — at least one source is unmatched
    2  — argument/IO error
"""

import argparse
import sys
from pathlib import Path

import cv2

from content_matcher import (
    image_files,
    get_features,
    inlier_count,
    compute_hist,
    hist_similarity,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Find source images with no matching crop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    p.add_argument("crops_dir", help="Directory of cropped/rescaled images")
    p.add_argument("sources_dir", help="Directory of original source images")
    p.add_argument(
        "--min-matches",
        type=int,
        default=8,
        help="Minimum RANSAC inliers to call a crop matched (default: 8). "
        "True matches typically produce 20–200+ inliers. "
        "Raise to reduce false positives; lower to catch weak matches.",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold (default: 0.75). "
        "Lower = stricter descriptor matching, fewer candidates for RANSAC.",
    )
    p.add_argument(
        "--ransac-thresh",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold in pixels (default: 5.0). "
        "Lower = stricter geometric fit.",
    )
    p.add_argument(
        "--nfeatures",
        type=int,
        default=2000,
        help="Max SIFT features per image (default: 2000).",
    )
    p.add_argument(
        "--hist-prefilter",
        type=float,
        default=0.0,
        metavar="THRESH",
        help="Skip SIFT matching when HSV histogram similarity < THRESH "
        "(0 = disabled; try 0.2 to speed up large datasets).",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="Cache keypoints+descriptors in SOURCES_DIR/.feature_cache/ "
        "to speed up repeated runs.",
    )
    p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write unmatched source paths to FILE, one per line.",
    )
    p.add_argument(
        "--names-only",
        action="store_true",
        help="Print only filenames (not full paths) in the results.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show inlier counts per pair.",
    )
    args = p.parse_args()

    crops_dir = Path(args.crops_dir).resolve()
    sources_dir = Path(args.sources_dir).resolve()

    for d, label in [(crops_dir, "crops"), (sources_dir, "sources")]:
        if not d.is_dir():
            print(f"error: {label} directory not found: {d}", file=sys.stderr)
            sys.exit(2)

    crop_files = image_files(crops_dir)
    source_files = image_files(sources_dir)

    print(f"Crops:   {len(crop_files):>5}  {crops_dir}")
    print(f"Sources: {len(source_files):>5}  {sources_dir}")

    if not crop_files:
        print("\nNo crops found — every source image counts as unmatched.")
        for f in source_files:
            print(f"  {f.name if args.names_only else f}")
        sys.exit(1 if source_files else 0)

    if not source_files:
        print("\nNo source images found.")
        sys.exit(0)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=args.nfeatures)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        detector_name = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=args.nfeatures)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        detector_name = "ORB"

    print(
        f"\nDetector: {detector_name}  nfeatures={args.nfeatures}  "
        f"min-inliers={args.min_matches}  ratio={args.ratio}  "
        f"ransac={args.ransac_thresh}px"
    )

    cache_dir = (sources_dir / ".feature_cache") if args.cache else None

    # ------------------------------------------------------------------
    # Step 1: extract features from all crops
    # ------------------------------------------------------------------
    print(f"\nExtracting features from {len(crop_files)} crops...")
    w = len(str(len(crop_files)))
    crop_data = []
    for i, path in enumerate(crop_files, 1):
        kps, des = get_features(path, detector, cache_dir)
        n = 0 if des is None else len(des)
        print(f"  [{i:{w}}/{len(crop_files)}] {path.name}  ({n} keypoints)")
        crop_data.append((path, kps, des))

    # ------------------------------------------------------------------
    # Step 2 (optional): precompute histograms for fast pre-filter
    # ------------------------------------------------------------------
    crop_hists: list | None = None
    src_hists: dict | None = None
    if args.hist_prefilter > 0:
        print("\nPrecomputing histograms for pre-filter...")
        crop_hists = [compute_hist(p) for p, _, _ in crop_data]
        src_hists = {p: compute_hist(p) for p in source_files}

    # ------------------------------------------------------------------
    # Step 3: match each source against all crops
    # ------------------------------------------------------------------
    print(f"\nChecking {len(source_files)} source images...")
    w2 = len(str(len(source_files)))
    unmatched: list[Path] = []

    for i, src_path in enumerate(source_files, 1):
        kp_src, des_src = get_features(src_path, detector, cache_dir)

        matched = False
        matched_by: Path | None = None
        best_score = 0

        if des_src is not None:
            for j, (crop_path, kp_crop, des_crop) in enumerate(crop_data):
                if args.hist_prefilter > 0:
                    sim = hist_similarity(crop_hists[j], src_hists[src_path])
                    if sim < args.hist_prefilter:
                        if args.verbose:
                            print(f"      skip {crop_path.name}  hist={sim:.2f}")
                        continue

                n = inlier_count(
                    kp_crop,
                    des_crop,
                    kp_src,
                    des_src,
                    matcher,
                    args.ratio,
                    args.ransac_thresh,
                )
                if args.verbose:
                    print(f"      {crop_path.name}  →  {n} inliers")
                if n > best_score:
                    best_score = n
                if n >= args.min_matches:
                    matched = True
                    matched_by = crop_path
                    break

        tag = "matched  " if matched else "UNMATCHED"
        detail = (
            f"  ← {matched_by.name} ({best_score} inliers)"
            if (matched and args.verbose)
            else (f"  (best: {best_score} inliers)" if args.verbose else "")
        )
        print(f"  [{i:{w2}}/{len(source_files)}] {tag}  {src_path.name}{detail}")

        if not matched:
            unmatched.append(src_path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"{len(unmatched)} of {len(source_files)} sources have no matching crop")
    print("─" * 60)

    if unmatched:
        print("\nUnmatched sources:")
        for path in unmatched:
            label = path.name if args.names_only else path
            print(f"  {label}")
    else:
        print("\nAll source images have at least one matching crop.")

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            for path in unmatched:
                label = path.name if args.names_only else path
                f.write(f"{label}\n")
        print(f"\nSaved to: {out}")

    sys.exit(0 if not unmatched else 1)


if __name__ == "__main__":
    main()
