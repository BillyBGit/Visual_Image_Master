#!/usr/bin/env python3
"""
img_sort_cli.py — command-line (no GUI) tools for bulk photo-library cleanup.
Every subcommand recurses over whole directory trees.

rotate — fix sideways photos from cameras without a tilt sensor
----------------------------------------------------------------
Three steps; nothing is modified until ``apply``:

  # 1. Analyse.  Writes a CSV report (and an embedding cache next to it so
  #    re-runs are fast).  Optionally builds a review folder of symlinks.
  python img_sort_cli.py rotate scan /photos --report rot.csv --review-dir /tmp/rot_review

  # 2. (optional) Open the review folder in img_sort.  Images sit in
  #    cw/ ccw/ upright/ according to the suggestion; drag any wrong ones to
  #    the right folder, delete a link to leave that image unchanged.

  # 3. Apply confident rotations (+ your review decisions).  Writes an undo log.
  python img_sort_cli.py rotate apply rot.csv --review-dir /tmp/rot_review

  # Undo
  python img_sort_cli.py rotate undo rotate_applied_20260927-141500.csv

Rotation is lossless: only the EXIF Orientation tag is written (via exiftool,
file modification times preserved).  Pixels are never re-encoded.

Report actions
--------------
  rotate   confident — ``apply`` will rotate it
  review   uncertain — only changed with apply --accept-review, if you place
           it via --review-dir, or if you edit the CSV (action=rotate,
           suggest=cw/ccw/180).  Most suggestions here are still right;
           the rest are mostly top-down shots with no real "up".
  keep     confidently upright
  tagged   already has a camera-written Orientation tag; left alone
           (use --include-tagged to re-check these)
  error    could not be read

Exit codes: 0 success, 1 some files failed, 2 argument/IO error.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import orientation as ori

REPORT_FIELDS = [
    "id",
    "path",
    "action",
    "suggest",
    "margin",
    "exif_orientation",
    "width",
    "height",
    "score_upright",
    "score_cw",
    "score_180",
    "score_ccw",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_orientation(path):
    """EXIF Orientation value (int) or None; raises on unreadable files."""
    return _read_header(path)[0]


def _read_header(path):
    """(EXIF Orientation or None, stored width, stored height) — header only."""
    from PIL import Image

    with Image.open(path) as im:
        return im.getexif().get(0x0112), im.width, im.height


def _cache_key(path):
    st = os.stat(path)
    # size is deliberately excluded: writing the Orientation tag changes the
    # file size but not the pixels (and exiftool -P keeps the mtime)
    return f"{path}|{st.st_mtime_ns}"


def _load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    d = np.load(cache_path, allow_pickle=False)
    return dict(zip(d["keys"].tolist(), d["emb"]))


def _save_cache(cache_path, cache):
    tmp = cache_path + ".tmp.npz"
    keys = list(cache)
    np.savez(tmp, keys=np.array(keys), emb=np.stack([cache[k] for k in keys]))
    os.replace(tmp, cache_path)


def _require_exiftool():
    if shutil.which("exiftool") is None:
        print("error: exiftool not found (sudo apt install libimage-exiftool-perl)")
        sys.exit(2)


def _exiftool_set_orientation(paths, value):
    """Set EXIF Orientation on many files in one exiftool run.
    Returns the number of files exiftool reports as updated."""
    if not paths:
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False) as f:
        f.write("\n".join(paths) + "\n")
        argfile = f.name
    try:
        proc = subprocess.run(
            [
                "exiftool",
                "-n",
                "-P",
                "-overwrite_original",
                f"-Orientation={value}",
                "-charset",
                "filename=utf8",
                "-@",
                argfile,
            ],
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(argfile)
    for line in proc.stderr.splitlines():
        print(f"  exiftool: {line}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.endswith("image files updated"):
            return int(line.split()[0])
    return 0


# ---------------------------------------------------------------------------
# rotate scan
# ---------------------------------------------------------------------------


def cmd_rotate_scan(args):
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: not a directory: {root}")
        return 2
    report = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report), exist_ok=True)
    cache_path = os.path.splitext(report)[0] + ".emb.npz"
    candidates = (0, 1, 2, 3) if args.allow_180 else (0, 1, 3)

    paths = list(ori.find_images(root))
    print(f"Found {len(paths)} images under {root}")
    if not paths:
        return 0

    # --- EXIF orientation + stored size (header read only) ---
    def safe_header(p):
        try:
            return _read_header(p)
        except Exception:
            return "error", 0, 0

    with ThreadPoolExecutor(args.workers) as ex:
        headers = list(ex.map(safe_header, paths))
    orients = [h[0] for h in headers]

    # --- embeddings (cached) ---
    cache = _load_cache(cache_path)
    keys = [_cache_key(p) for p in paths]
    todo = [
        i
        for i, (k, o) in enumerate(zip(keys, orients))
        if k not in cache and o != "error"
    ]
    print(f"Embedding {len(todo)} images ({len(paths) - len(todo)} cached/skipped)")
    if todo:
        embedder = ori.OrientationEmbedder(args.model)

        def safe_load(i):
            try:
                return i, ori.load_square(paths[i])
            except Exception as e:
                print(f"\n  unreadable: {paths[i]} ({e})")
                return i, None

        t0 = time.time()
        done = 0
        last_save = 0
        with ThreadPoolExecutor(args.workers) as ex:
            for start in range(0, len(todo), args.batch):
                batch = list(ex.map(safe_load, todo[start : start + args.batch]))
                good = [(i, sq) for i, sq in batch if sq is not None]
                for i, sq in batch:
                    if sq is None:
                        orients[i] = "error"
                if good:
                    emb = embedder.embed(np.stack([sq for _, sq in good]))
                    for (i, _), e in zip(good, emb):
                        cache[keys[i]] = e
                done += len(batch)
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"\r  {done}/{len(todo)}  {rate:.1f} img/s  "
                    f"eta {(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min   ",
                    end="",
                    flush=True,
                )
                if done - last_save >= 2000:  # resumable if interrupted
                    _save_cache(cache_path, cache)
                    last_save = done
        print()
        _save_cache(cache_path, cache)

    ok = [i for i, o in enumerate(orients) if o != "error"]
    emb = np.stack([cache[keys[i]] for i in ok])
    exif = [orients[i] for i in ok]

    # --- classifier ---
    known = np.array(
        [ori.EXIF_TO_K.get(o, -1) if o not in (None, 1) else -1 for o in exif]
    )
    if args.probe and os.path.exists(args.probe):
        clf = ori.UprightClassifier.load(args.probe)
        print(f"Loaded classifier from {args.probe}")
    else:
        if len(ok) < 200:
            print(
                f"warning: only {len(ok)} images to self-train on; results may be "
                "unreliable.  Consider --probe with a classifier saved from a "
                "larger collection."
            )
        clf = ori.self_train(emb, known, candidates=candidates)
        probe_out = args.probe or os.path.splitext(report)[0] + ".probe.npz"
        clf.save(probe_out)
        print(f"Trained classifier on {len(ok)} images -> {probe_out}")

    scores = clf.scores(emb)
    best, margin = ori.decide(scores, candidates)

    # --- decisions ---
    rows = []
    counts = {}
    pos = {i: j for j, i in enumerate(ok)}
    for i, p in enumerate(paths):
        o = orients[i]
        _, width, height = headers[i]
        row = {
            "id": i,
            "path": p,
            "exif_orientation": "" if o is None else o,
            "width": width,
            "height": height,
        }
        if o == "error":
            row.update(action="error", suggest="")
        else:
            j = pos[i]
            k, m = int(best[j]), float(margin[j])
            row.update(
                suggest=ori.K_LABELS[k],
                margin=f"{m:.3f}",
                **{f"score_{ori.K_LABELS[r]}": f"{scores[j, r]:.3f}" for r in range(4)},
            )
            tagged = o not in (None, 1)
            if tagged and not args.include_tagged:
                row["action"] = "tagged"
            elif k == 0:
                row["action"] = "keep" if m >= args.keep_margin else "review"
            else:
                # portrait-shaped pixels were almost always rotated by some
                # earlier software already, so demand much more confidence
                need = args.portrait_margin if height > width else args.rotate_margin
                row["action"] = "rotate" if m >= need else "review"
            if tagged and args.include_tagged and row["action"] != "review":
                # stored pixels need k; the tag already says EXIF_TO_K[o]
                row["action"] = "keep" if ori.EXIF_TO_K.get(o) == k else "rotate"
        counts[row["action"]] = counts.get(row["action"], 0) + 1
        rows.append(row)

    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, REPORT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # --- review folder ---
    if args.review_dir:
        rdir = Path(args.review_dir)
        if rdir.exists() and any(rdir.iterdir()):
            print(f"error: review dir {rdir} is not empty; choose a new one")
            return 2
        n = 0
        for r in rows:
            if r["action"] != "review":
                continue
            sub = rdir / r["suggest"]
            sub.mkdir(parents=True, exist_ok=True)
            os.symlink(r["path"], sub / f"{r['id']:06d}__{os.path.basename(r['path'])}")
            n += 1
        for label in ("upright", "cw", "ccw") + (("180",) if args.allow_180 else ()):
            (rdir / label).mkdir(parents=True, exist_ok=True)
        print(f"Review folder: {rdir} ({n} links)")

    rot = [r for r in rows if r["action"] == "rotate"]
    by_dir = {}
    for r in rot:
        by_dir[r["suggest"]] = by_dir.get(r["suggest"], 0) + 1
    print(f"\nReport: {report}")
    for a in ("rotate", "review", "keep", "tagged", "error"):
        if counts.get(a):
            extra = (
                f"  ({', '.join(f'{v} {k}' for k, v in by_dir.items())})"
                if a == "rotate"
                else ""
            )
            print(f"  {a:7s} {counts[a]:6d}{extra}")
    print("\nNothing changed yet. Next: img_sort_cli.py rotate apply " + args.report)
    return 0


# ---------------------------------------------------------------------------
# rotate apply / undo
# ---------------------------------------------------------------------------


def _review_decisions(review_dir):
    """Map report id -> k from where the user left each symlink."""
    decisions = {}
    for label, k in ori.LABEL_TO_K.items():
        sub = Path(review_dir) / label
        if not sub.is_dir():
            continue
        for link in sub.iterdir():
            head = link.name.split("__", 1)[0]
            if head.isdigit():
                decisions[int(head)] = k
    return decisions


def cmd_rotate_apply(args):
    _require_exiftool()
    with open(args.report, newline="") as f:
        rows = list(csv.DictReader(f))

    targets = {}  # path -> k
    for r in rows:
        wanted = r["action"] == "rotate" or (
            args.accept_review and r["action"] == "review"
        )
        if wanted and r["suggest"] in ori.LABEL_TO_K:
            targets[r["path"]] = ori.LABEL_TO_K[r["suggest"]]
    n_auto = len(targets)
    if args.review_dir:
        by_id = {int(r["id"]): r for r in rows}
        for rid, k in _review_decisions(args.review_dir).items():
            if rid in by_id:
                targets[by_id[rid]["path"]] = k
        print(f"Review folder decisions: {len(targets) - n_auto} added/changed")

    changes = []  # (path, old, new)
    for path, k in targets.items():
        if not os.path.exists(path):
            print(f"  missing, skipped: {path}")
            continue
        old = _read_orientation(path)
        new = ori.K_TO_EXIF[k]
        if (old or 1) != new:
            changes.append((path, "" if old is None else old, new))

    by_value = {}
    for path, _, new in changes:
        by_value.setdefault(new, []).append(path)
    summary = ", ".join(
        f"{len(v)} -> {ori.K_LABELS[ori.EXIF_TO_K[k]]}"
        for k, v in sorted(by_value.items())
    )
    print(f"{len(changes)} files to update ({summary or 'none'})")
    if args.dry_run or not changes:
        return 0

    log = os.path.abspath(
        args.log or f"rotate_applied_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    with open(log, "w", newline="") as f:  # written first so undo is always possible
        w = csv.writer(f)
        w.writerow(["path", "old_orientation", "new_orientation"])
        w.writerows(changes)

    updated = sum(_exiftool_set_orientation(ps, v) for v, ps in by_value.items())
    print(f"Updated {updated}/{len(changes)} files.  Undo log: {log}")
    return 0 if updated == len(changes) else 1


def cmd_rotate_undo(args):
    _require_exiftool()
    with open(args.log, newline="") as f:
        rows = list(csv.DictReader(f))
    by_value = {}
    for r in rows:
        by_value.setdefault(int(r["old_orientation"] or 1), []).append(r["path"])
    updated = sum(_exiftool_set_orientation(ps, v) for v, ps in by_value.items())
    print(f"Restored {updated}/{len(rows)} files")
    return 0 if updated == len(rows) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        prog="img_sort_cli.py",
        description="Command-line photo-library tools (recursive, no GUI).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    rot = sub.add_parser("rotate", help="detect and fix sideways photos")
    rsub = rot.add_subparsers(dest="rotate_cmd", required=True)

    s = rsub.add_parser("scan", help="analyse a directory tree, write a report")
    s.add_argument("root", help="directory to scan recursively")
    s.add_argument("--report", default="rotate_report.csv", help="output CSV")
    s.add_argument("--review-dir", help="create symlinks for uncertain images here")
    s.add_argument(
        "--rotate-margin",
        type=float,
        default=1.0,
        help="min confidence margin to auto-rotate (default 1.0; lower = more "
        "automatic, more mistakes)",
    )
    s.add_argument(
        "--portrait-margin",
        type=float,
        default=3.0,
        help="min margin to auto-rotate an image whose stored pixels are "
        "already portrait-shaped (default 3.0)",
    )
    s.add_argument(
        "--keep-margin",
        type=float,
        default=0.25,
        help="images judged upright with a margin below this go to review",
    )
    s.add_argument("--allow-180", action="store_true", help="also consider upside-down")
    s.add_argument(
        "--include-tagged",
        action="store_true",
        help="re-check images that already have a camera Orientation tag",
    )
    s.add_argument("--probe", help="load (or save) the trained classifier here")
    s.add_argument("--model", default=ori.DEFAULT_MODEL, help="CLIP model id")
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--workers", type=int, default=8, help="image-decoding threads")
    s.set_defaults(func=cmd_rotate_scan)

    a = rsub.add_parser("apply", help="write Orientation tags from a report")
    a.add_argument("report")
    a.add_argument("--review-dir", help="also apply decisions from a review folder")
    a.add_argument(
        "--accept-review",
        action="store_true",
        help="also apply the suggestion for every 'review' row (review-folder "
        "placements still take precedence)",
    )
    a.add_argument("--log", help="undo log path (default rotate_applied_<time>.csv)")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(func=cmd_rotate_apply)

    u = rsub.add_parser("undo", help="restore Orientation tags from an apply log")
    u.add_argument("log")
    u.set_defaults(func=cmd_rotate_undo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
