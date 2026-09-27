"""Tests for orientation.py and the `rotate` subcommand of img_sort_cli.py.

torch / transformers are never loaded: embeddings are synthetic numpy arrays and
exiftool calls are mocked.
"""

import csv
import os
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import img_sort_cli
import orientation as ori

# ---------------------------------------------------------------------------
# orientation.py
# ---------------------------------------------------------------------------


def test_k_exif_mappings_round_trip():
    assert ori.K_TO_EXIF == {0: 1, 1: 6, 2: 3, 3: 8}
    for k, v in ori.K_TO_EXIF.items():
        assert ori.EXIF_TO_K[v] == k
    for k, label in ori.K_LABELS.items():
        assert ori.LABEL_TO_K[label] == k


def test_decide_picks_best_candidate_and_margin():
    scores = np.array(
        [
            [5.0, 1.0, 9.0, 2.0],  # 180 highest but not a candidate -> 0
            [0.0, 3.0, 0.0, 1.0],  # cw
            [0.0, 1.0, 0.0, 4.0],  # ccw
        ]
    )
    best, margin = ori.decide(scores, candidates=(0, 1, 3))
    assert best.tolist() == [0, 1, 3]
    assert margin.tolist() == pytest.approx([3.0, 2.0, 3.0])


def test_decide_with_180_allowed():
    best, _ = ori.decide(np.array([[5.0, 1.0, 9.0, 2.0]]), candidates=(0, 1, 2, 3))
    assert best.tolist() == [2]


def test_classifier_scores_save_load(tmp_path):
    w = np.array([1.0, -1.0], dtype=np.float32)
    clf = ori.UprightClassifier(w, 0.5)
    emb = np.array([[[1, 0], [0, 1], [0, 0], [1, 1]]], dtype=np.float16)
    assert clf.scores(emb).tolist() == [[1.5, -0.5, 0.5, 0.5]]
    path = tmp_path / "probe.npz"
    clf.save(path)
    loaded = ori.UprightClassifier.load(path)
    assert loaded.scores(emb).tolist() == clf.scores(emb).tolist()


def test_find_images_recurses_filters_and_skips_hidden(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / ".hidden").mkdir()
    for p in ["x.JPG", "a/y.jpeg", "a/b/z.tif", "a/notes.txt", ".hidden/h.jpg"]:
        (tmp_path / p).write_bytes(b"")
    found = [os.path.relpath(p, tmp_path) for p in ori.find_images(tmp_path)]
    assert sorted(found) == ["a/b/z.tif", "a/y.jpeg", "x.JPG"]


def test_load_square_keeps_stored_pixels_and_crops_centre(tmp_path):
    # landscape image: left half red, right half blue, with Orientation=6 tag
    arr = np.zeros((300, 600, 3), dtype=np.uint8)
    arr[:, :300] = (255, 0, 0)
    arr[:, 300:] = (0, 0, 255)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6
    path = tmp_path / "t.jpg"
    im.save(path, exif=exif, quality=95)

    sq = ori.load_square(str(path), size=64)
    assert sq.shape == (64, 64, 3)
    # tag must NOT be applied: red stays on the left, blue on the right
    assert sq[32, 5, 0] > 200 and sq[32, 5, 2] < 60
    assert sq[32, 58, 2] > 200 and sq[32, 58, 0] < 60


# ---------------------------------------------------------------------------
# img_sort_cli rotate apply / undo
# ---------------------------------------------------------------------------


def _write_report(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, img_sort_cli.REPORT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in img_sort_cli.REPORT_FIELDS})


def _jpeg(path):
    Image.new("RGB", (8, 4)).save(path)
    return str(path)


def test_review_decisions_reads_folder_placement(tmp_path):
    for label, name in [("cw", "000003__a.jpg"), ("upright", "000007__b.jpg")]:
        (tmp_path / label).mkdir()
        (tmp_path / label / name).write_bytes(b"")
    (tmp_path / "ccw").mkdir()
    (tmp_path / "ccw" / "junk.jpg").write_bytes(b"")  # no id prefix -> ignored
    assert img_sort_cli._review_decisions(tmp_path) == {3: 1, 7: 0}


def test_apply_rotates_confident_rows_and_review_choices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _jpeg(tmp_path / "a.jpg")  # action rotate cw
    b = _jpeg(tmp_path / "b.jpg")  # review, user moved to ccw
    c = _jpeg(tmp_path / "c.jpg")  # review, link deleted -> unchanged
    d = _jpeg(tmp_path / "d.jpg")  # keep
    report = tmp_path / "rot.csv"
    _write_report(
        report,
        [
            {"id": 0, "path": a, "action": "rotate", "suggest": "cw"},
            {"id": 1, "path": b, "action": "review", "suggest": "cw"},
            {"id": 2, "path": c, "action": "review", "suggest": "ccw"},
            {"id": 3, "path": d, "action": "keep", "suggest": "upright"},
        ],
    )
    review = tmp_path / "review"
    (review / "ccw").mkdir(parents=True)
    (review / "ccw" / "000001__b.jpg").write_bytes(b"")

    calls = {}

    def fake_set(paths, value):
        calls[value] = sorted(paths)
        return len(paths)

    with (
        patch("img_sort_cli._require_exiftool"),
        patch("img_sort_cli._exiftool_set_orientation", side_effect=fake_set),
    ):
        rc = img_sort_cli.main(
            [
                "rotate",
                "apply",
                str(report),
                "--review-dir",
                str(review),
                "--log",
                "u.csv",
            ]
        )

    assert rc == 0
    assert calls == {6: [a], 8: [b]}
    with open(tmp_path / "u.csv", newline="") as f:
        log = {r["path"]: r for r in csv.DictReader(f)}
    assert set(log) == {a, b}
    assert log[a]["old_orientation"] == "" and log[a]["new_orientation"] == "6"


def test_apply_dry_run_changes_nothing(tmp_path):
    a = _jpeg(tmp_path / "a.jpg")
    report = tmp_path / "rot.csv"
    _write_report(report, [{"id": 0, "path": a, "action": "rotate", "suggest": "cw"}])
    with (
        patch("img_sort_cli._require_exiftool"),
        patch("img_sort_cli._exiftool_set_orientation") as set_mock,
    ):
        rc = img_sort_cli.main(["rotate", "apply", str(report), "--dry-run"])
    assert rc == 0
    set_mock.assert_not_called()


def test_undo_restores_old_values(tmp_path):
    log = tmp_path / "u.csv"
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "old_orientation", "new_orientation"])
        w.writerows([["/p/a.jpg", "", 6], ["/p/b.jpg", "1", 8], ["/p/c.jpg", "6", 1]])
    calls = {}

    def fake_set(paths, value):
        calls[value] = sorted(paths)
        return len(paths)

    with (
        patch("img_sort_cli._require_exiftool"),
        patch("img_sort_cli._exiftool_set_orientation", side_effect=fake_set),
    ):
        rc = img_sort_cli.main(["rotate", "undo", str(log)])
    assert rc == 0
    assert calls == {1: ["/p/a.jpg", "/p/b.jpg"], 6: ["/p/c.jpg"]}


def test_apply_accept_review_takes_suggestions_but_folder_wins(tmp_path):
    a = _jpeg(tmp_path / "a.jpg")  # review cw, accepted as-is
    b = _jpeg(tmp_path / "b.jpg")  # review cw, user put in upright -> unchanged
    report = tmp_path / "rot.csv"
    _write_report(
        report,
        [
            {"id": 0, "path": a, "action": "review", "suggest": "cw"},
            {"id": 1, "path": b, "action": "review", "suggest": "cw"},
        ],
    )
    review = tmp_path / "review"
    (review / "upright").mkdir(parents=True)
    (review / "upright" / "000001__b.jpg").write_bytes(b"")
    calls = {}

    def fake_set(paths, value):
        calls[value] = sorted(paths)
        return len(paths)

    with (
        patch("img_sort_cli._require_exiftool"),
        patch("img_sort_cli._exiftool_set_orientation", side_effect=fake_set),
    ):
        rc = img_sort_cli.main(
            [
                "rotate",
                "apply",
                str(report),
                "--accept-review",
                "--review-dir",
                str(review),
                "--log",
                str(tmp_path / "u.csv"),
            ]
        )
    assert rc == 0
    assert calls == {6: [a]}
