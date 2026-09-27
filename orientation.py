"""
Content-based photo orientation detection (no Qt).

Old compact cameras had no tilt sensor, so every portrait shot is stored
sideways with an EXIF Orientation of 1 ("normal").  This module works out
which way up each image should be from its pixels alone.

Method
------
1. Each image is embedded with CLIP at all four 90° rotations.
2. A small logistic-regression "is this upright?" classifier is trained on the
   collection itself (self-supervised): every image is assumed upright, which
   is true for the large majority, and the three other rotations are negatives.
   Images the first model is confident are sideways get relabelled and the
   model is retrained.  Images whose camera already wrote a real Orientation
   tag are used as correctly-labelled training data.
3. The rotation with the highest "upright" score wins; the gap to the
   runner-up is the confidence margin.

Rotations are expressed as ``k`` = number of 90° *clockwise* turns needed to
make the stored pixels upright, and are applied losslessly by writing the
EXIF Orientation tag (see ``K_TO_EXIF``).

Accuracy on a hand-labelled sample of 2003-2008 Coolpix photos: ~97% overall,
~98% among images above the default margin threshold.
"""

import os

import numpy as np

# k (clockwise quarter-turns needed) -> EXIF Orientation value, and back
K_TO_EXIF = {0: 1, 1: 6, 2: 3, 3: 8}
EXIF_TO_K = {v: k for k, v in K_TO_EXIF.items()}
K_LABELS = {0: "upright", 1: "cw", 2: "180", 3: "ccw"}
LABEL_TO_K = {v: k for k, v in K_LABELS.items()}

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff"}

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def find_images(root):
    """Recursively yield image paths under ``root`` (sorted, hidden dirs skipped)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def load_square(path, size=224):
    """Decode ``path`` as stored (EXIF orientation NOT applied), resize the
    short side to ``size`` and centre-crop to a square uint8 RGB array.

    A centre square crop commutes with 90° rotations, so the four rotated
    views can be produced later from this one array.
    """
    from PIL import Image

    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))  # fast JPEG DCT downscale
        im = im.convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        im = im.resize(
            (max(size, round(w * scale)), max(size, round(h * scale))),
            Image.BICUBIC,
        )
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        return np.asarray(im, dtype=np.uint8)


class OrientationEmbedder:
    """CLIP image embeddings of an image at all four rotations."""

    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        import torch
        from transformers import CLIPModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            CLIPModel.from_pretrained(model_name).to(self.device, self.dtype).eval()
        )
        self._mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def embed(self, squares):
        """``squares``: (n, S, S, 3) uint8.  Returns (n, 4, D) float16 L2-normalised
        embeddings where ``[:, k]`` is the image turned ``k`` quarter-turns
        clockwise."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(squares)).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self._mean) / self._std
        # torch.rot90 with k>0 turns counter-clockwise; negative k = clockwise
        views = torch.stack([torch.rot90(x, -k, dims=(2, 3)) for k in range(4)], 1)
        n = views.shape[0]
        with torch.no_grad():
            feats = self.model.get_image_features(
                pixel_values=views.reshape(n * 4, *views.shape[2:]).to(self.dtype)
            )
            feats = getattr(feats, "pooler_output", feats)  # transformers >= 5
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.reshape(n, 4, -1).half().cpu().numpy()


class UprightClassifier:
    """Linear 'is this view upright?' classifier over CLIP embeddings."""

    def __init__(self, weight=None, bias=0.0):
        self.weight = weight
        self.bias = bias

    def fit(self, emb, ks, l2=1e-3, device=None):
        """Train from ``emb`` (n, 4, D) where ``ks[i]`` is the assumed correct
        rotation of image i: view ``(ks[i] + j) % 4`` is a positive when j == 0
        and a negative otherwise."""
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n = len(emb)
        idx = np.arange(n)
        xs = [emb[idx, (ks + j) % 4] for j in range(4)]
        x = torch.from_numpy(np.concatenate(xs).astype(np.float32)).to(device)
        y = torch.cat(
            [torch.ones(n) if j == 0 else torch.zeros(n) for j in range(4)]
        ).to(device)
        w = torch.zeros(x.shape[1], device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
        # weight positives 3x so both classes count equally
        sample_w = torch.where(y > 0, 3.0, 1.0)
        # embeddings are unit-norm with small components; scale logits up so a
        # modest weight vector can separate them
        scale = 10.0

        def closure():
            opt.zero_grad()
            logits = scale * (x @ w) + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=sample_w
            )
            loss = loss + l2 * (w @ w)
            loss.backward()
            return loss

        opt.step(closure)
        self.weight = (scale * w).detach().cpu().numpy()
        self.bias = float(b.detach().cpu())
        return self

    def scores(self, emb):
        """(n, 4) upright logits for each rotation."""
        return emb.astype(np.float32) @ self.weight + self.bias

    def save(self, path):
        np.savez(path, weight=self.weight, bias=self.bias)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["weight"], float(d["bias"]))


def self_train(emb, known_k=None, rounds=2, relabel_margin=1.0, candidates=(0, 1, 3)):
    """Train an ``UprightClassifier`` on a collection without labels.

    ``known_k``: optional int array, -1 where unknown, else the trusted k for
    that image (e.g. from a camera-written EXIF Orientation tag).
    """
    n = len(emb)
    known = np.full(n, -1) if known_k is None else np.asarray(known_k)
    ks = np.where(known >= 0, known, 0)
    clf = UprightClassifier().fit(emb, ks)
    for _ in range(rounds - 1):
        best, margin = decide(clf.scores(emb), candidates)
        ks = np.where(known >= 0, known, np.where(margin > relabel_margin, best, 0))
        clf = UprightClassifier().fit(emb, ks)
    return clf


def decide(scores, candidates=(0, 1, 3)):
    """Pick the best rotation among ``candidates``.

    Returns (best_k, margin) arrays; margin = best score - runner-up score.
    """
    cands = np.asarray(candidates)
    sub = scores[:, cands]
    order = np.argsort(-sub, axis=1)
    best = cands[order[:, 0]]
    rows = np.arange(len(sub))
    margin = sub[rows, order[:, 0]] - sub[rows, order[:, 1]]
    return best, margin
