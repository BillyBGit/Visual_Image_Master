#!/usr/bin/env python3
"""
sync_captions.py — Rename background-removed images to match their originals,
then copy caption .txt files across.

Scenario
--------
  DIR1  background-removed images (often .png, possibly with different base names)
  DIR2  original images that already have matching .txt caption files

The script pairs each DIR1 file to its best-matching DIR2 file by filename
similarity, proposes a rename (keeping DIR1's own extension), and optionally
copies the corresponding .txt caption from DIR2 into DIR1.

Dry-run by default — no files are changed.  Pass --apply to execute.

Usage
-----
  python sync_captions.py DIR1 DIR2 [options]

  # Preview what would happen
  python sync_captions.py ./no_bg ./originals

  # Execute renames and caption copies
  python sync_captions.py ./no_bg ./originals --apply

  # Rename only, skip caption copying
  python sync_captions.py ./no_bg ./originals --apply --no-captions

  # Lower the similarity threshold if some expected matches are missed
  python sync_captions.py ./no_bg ./originals --min-score 0.3

Exit codes
----------
  0  all DIR1 files matched and (if --apply) operations succeeded
  1  one or more DIR1 files could not be matched
  2  argument / IO error
"""

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".avif",
    ".gif",
}


# ---------------------------------------------------------------------------
# Filename similarity (same algorithm as pair_selector_window.py)
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    """Score [0, 1] for how similar two base-names are."""
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


def _best_match(base1: str, candidates: list[tuple[str, Path]]) -> tuple[float, int]:
    """Return (best_score, best_index) for base1 against a list of (base, path)."""
    best_score, best_idx = 0.0, -1
    for i, (base2, _) in enumerate(candidates):
        s = _similarity(base1, base2)
        if s > best_score:
            best_score, best_idx = s, i
    return best_score, best_idx


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------


def image_files(directory: Path) -> list[Path]:
    return sorted(
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def build_matches(
    files1: list[Path],
    files2: list[Path],
    min_score: float,
) -> tuple[list[tuple[Path, Path, float]], list[Path], list[Path]]:
    """
    Greedily match each DIR1 file to the best unmatched DIR2 file.

    Returns
    -------
    matched   : [(dir1_path, dir2_path, score), ...]
    unmatched1: DIR1 files with no good match
    unmatched2: DIR2 files left over after matching
    """
    candidates = [(f.stem, f) for f in files2]
    used = set()

    matched: list[tuple[Path, Path, float]] = []
    unmatched1: list[Path] = []

    for f1 in files1:
        remaining = [
            (base, path) for i, (base, path) in enumerate(candidates) if i not in used
        ]
        if not remaining:
            unmatched1.append(f1)
            continue

        # Map remaining indices back to original candidate indices for 'used' tracking
        remaining_indices = [i for i in range(len(candidates)) if i not in used]
        remaining_bases = [
            (candidates[i][0], candidates[i][1]) for i in remaining_indices
        ]

        base1 = f1.stem
        best_score, best_local_idx = _best_match(base1, remaining_bases)

        if best_local_idx == -1 or best_score < min_score:
            unmatched1.append(f1)
        else:
            global_idx = remaining_indices[best_local_idx]
            used.add(global_idx)
            matched.append((f1, candidates[global_idx][1], best_score))

    unmatched2 = [candidates[i][1] for i in range(len(candidates)) if i not in used]
    return matched, unmatched1, unmatched2


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


def _unique_path(path: Path) -> Path:
    """If path exists, append _N until a free name is found."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 1
    while True:
        candidate = path.with_name(f"{stem}_{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Rename background-removed images to match originals and copy captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    p.add_argument(
        "dir1", help="Directory of background-removed images (will be renamed)"
    )
    p.add_argument("dir2", help="Directory of originals with .txt caption files")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Execute renames and caption copies (default: dry-run only)",
    )
    p.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip copying .txt caption files (rename only)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        metavar="SCORE",
        help="Minimum filename similarity to accept a match [0–1] (default: 0.5). "
        "Lower if expected matches are being missed.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show similarity scores for each match.",
    )
    args = p.parse_args()

    dir1 = Path(args.dir1).resolve()
    dir2 = Path(args.dir2).resolve()

    for d, label in [(dir1, "DIR1"), (dir2, "DIR2")]:
        if not d.is_dir():
            print(f"error: {label} not found: {d}", file=sys.stderr)
            sys.exit(2)

    files1 = image_files(dir1)
    files2 = image_files(dir2)

    print(f"DIR1 (bg-removed):  {len(files1):>5}  {dir1}")
    print(f"DIR2 (with captions): {len(files2):>4}  {dir2}")
    print(f"Min similarity score: {args.min_score}")
    print()

    if not files1:
        print("No images found in DIR1.")
        sys.exit(0)
    if not files2:
        print("No images found in DIR2.")
        sys.exit(2)

    matched, unmatched1, unmatched2 = build_matches(files1, files2, args.min_score)

    # ------------------------------------------------------------------
    # Report matches
    # ------------------------------------------------------------------
    col = max((len(f1.name) for f1, _, _ in matched), default=20)
    col2 = max((len(f2.stem + f1.suffix) for f1, f2, _ in matched), default=20)

    print(f"{'Matched pairs':─<60}")
    rename_ops: list[tuple[Path, Path]] = []  # (old_path, new_path)
    caption_ops: list[tuple[Path, Path]] = []  # (src_txt, dst_txt)

    for f1, f2, score in matched:
        new_name = f2.stem + f1.suffix  # keep DIR1 extension (e.g. .png)
        new_path = dir1 / new_name
        src_txt = f2.with_suffix(".txt")
        dst_txt = new_path.with_suffix(".txt")

        has_caption = src_txt.exists()
        already_named = f1.name == new_name

        score_str = f"  [{score:.2f}]" if args.verbose else ""
        rename_tag = "(already correct)" if already_named else f"→ {new_name:{col2}}"
        caption_tag = "  +caption" if has_caption and not args.no_captions else ""

        print(f"  {f1.name:{col}}  {rename_tag}{caption_tag}{score_str}")

        if not already_named:
            rename_ops.append((f1, new_path))
        if has_caption and not args.no_captions:
            caption_ops.append((src_txt, dst_txt))

    print()

    if unmatched1:
        print(f"{'Unmatched in DIR1 (no rename)':─<60}")
        for f in unmatched1:
            print(f"  {f.name}")
        print()

    if unmatched2:
        print(f"{'Unmatched in DIR2 (no caption available)':─<60}")
        for f in unmatched2:
            txt = f.with_suffix(".txt")
            tag = "  [has caption]" if txt.exists() else ""
            print(f"  {f.name}{tag}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    already_ok = len(matched) - len(rename_ops)
    print("Summary:")
    print(f"  {len(matched)} matched  ({already_ok} already correctly named)")
    print(f"  {len(rename_ops)} rename(s) needed")
    print(f"  {len(caption_ops)} caption(s) to copy")
    print(f"  {len(unmatched1)} DIR1 file(s) unmatched")
    print()

    if not args.apply:
        print("Dry run — no changes made.  Pass --apply to execute.")
        sys.exit(0 if not unmatched1 else 1)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    errors = 0

    if rename_ops:
        print(f"Renaming {len(rename_ops)} file(s)...")
        for old_path, new_path in rename_ops:
            # If the target name already exists (different file), disambiguate
            if new_path.exists() and new_path != old_path:
                new_path = _unique_path(new_path)
                print(f"  conflict: renamed to {new_path.name}")
            try:
                old_path.rename(new_path)
                print(f"  {old_path.name}  →  {new_path.name}")
            except OSError as e:
                print(f"  ERROR renaming {old_path.name}: {e}", file=sys.stderr)
                errors += 1

    if caption_ops:
        print(f"\nCopying {len(caption_ops)} caption(s)...")
        for src_txt, dst_txt in caption_ops:
            # dst_txt path uses the NEW name (computed above from f2.stem + f1.suffix)
            # Re-derive in case the rename was disambiguated — for simplicity we
            # just check existence of the txt before writing.
            try:
                shutil.copy2(src_txt, dst_txt)
                print(f"  {src_txt.name}  →  {dst_txt.name}")
            except OSError as e:
                print(f"  ERROR copying {src_txt.name}: {e}", file=sys.stderr)
                errors += 1

    print()
    if errors:
        print(f"{errors} error(s) occurred.")
        sys.exit(1)
    else:
        print("Done.")
        sys.exit(0 if not unmatched1 else 1)


if __name__ == "__main__":
    main()
