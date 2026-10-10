"""
Face embedding and identity comparison, using OpenCV's YuNet detector +
SFace recognizer (both from the OpenCV Zoo) - used for identity-grouping
and reference-image similarity sorting.

No Qt dependency. Results are cached per image path, mirroring the other
detector caches (FaceDetector, ObjectDetector, SceneRecognizer,
QualityAssessor) in this project.
"""

import os

import cv2


class FaceEmbedder:
    """Detects the largest face in an image and extracts an SFace feature
    vector for identity comparison."""

    # SFace's own recommended cosine-similarity threshold for "same person"
    SAME_IDENTITY_THRESHOLD = 0.363

    def __init__(self):
        self._faces_cache = {}  # normalized path -> list of embeddings (largest first)
        self.detector = None
        self.recognizer = None
        self._load_models()

    def _load_models(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        detector_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        recognizer_path = os.path.join(
            models_dir, "face_recognition_sface_2021dec.onnx"
        )

        if not (os.path.exists(detector_path) and os.path.exists(recognizer_path)):
            print(
                "⚠ Face recognition model files not found. Run "
                "models/download_models.py to download them."
            )
            print("  Identity grouping and similarity sorting will be unavailable.")
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                detector_path, "", (320, 320), score_threshold=0.7
            )
            self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        except Exception as e:
            print(f"⚠ Could not load face recognition models: {e}")
            self.detector = None
            self.recognizer = None

    def is_available(self):
        return self.detector is not None and self.recognizer is not None

    def get_face_embeddings(self, image_path):
        """Return an SFace embedding for EVERY face detected in the image
        (largest first), or [] if unavailable, unreadable, or no face is
        detected. Cached.

        Matching only the single largest face misses the mark whenever
        the subject of interest isn't the largest bounding box in frame -
        two people can be nearly the same size in a shot, or a bystander
        closer to the camera can outsize the actual subject. ~10% of a
        sampled real photo collection had more than one detected face, in
        at least one case with two faces within 10% of each other's area
        - close enough that detector noise alone could flip which one
        "largest" picks. Embedding every face and letting the caller
        match against all of them removes that single-face guess.
        """
        cache_key = os.path.normpath(str(image_path))
        if cache_key in self._faces_cache:
            return self._faces_cache[cache_key]

        embeddings = []
        if self.is_available():
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    self.detector.setInputSize((w, h))
                    _, faces = self.detector.detect(img)
                    if faces is not None:
                        for face in sorted(
                            faces, key=lambda f: f[2] * f[3], reverse=True
                        ):
                            aligned = self.recognizer.alignCrop(img, face)
                            embeddings.append(self.recognizer.feature(aligned))
            except Exception:
                embeddings = []

        self._faces_cache[cache_key] = embeddings
        return embeddings

    def get_embedding(self, image_path):
        """Return the SFace embedding for the largest face in the image, or
        None if unavailable, unreadable, or no face is detected. Cached.

        Kept for callers that only care about a single face per image
        (identity clustering, single-image similarity); see
        get_face_embeddings for the multi-face version used by the
        reference-gallery similarity sorts.
        """
        embeddings = self.get_face_embeddings(image_path)
        return embeddings[0] if embeddings else None

    def compute_similarity(self, path_a, path_b):
        """Cosine similarity (roughly -1..1, higher = more alike) between
        whichever pair of detected faces (one per image) scores highest, or
        None if either image has no detectable face."""
        if not self.is_available():
            return None
        embs_a = self.get_face_embeddings(path_a)
        embs_b = self.get_face_embeddings(path_b)
        if not embs_a or not embs_b:
            return None
        best = None
        for emb_a in embs_a:
            score = self._best_match(emb_a, embs_b)
            if score is not None and (best is None or score > best):
                best = score
        return best

    def _best_match(self, emb, reference_embeddings):
        """Highest cosine similarity between emb and any one embedding in
        reference_embeddings, or None if reference_embeddings is empty."""
        best = None
        for ref_emb in reference_embeddings:
            score = float(
                self.recognizer.match(emb, ref_emb, cv2.FaceRecognizerSF_FR_COSINE)
            )
            if best is None or score > best:
                best = score
        return best

    def best_match_per_reference(self, candidate_embeddings, reference_embeddings):
        """For one candidate image's face embeddings (there may be more
        than one face in the photo) against EACH reference embedding
        individually, return a list of scores - one per reference
        embedding - of the candidate's own best-matching face against
        that one specific reference. The caller's overall similarity
        score is max() of this list - one call covers both "match every
        candidate face against every reference face" and "keep the score
        broken down per reference" for callers that want to inspect it
        (e.g. order_by_similarity_multi/_discriminative only need the
        max, but this is also how the per-reference breakdown got used
        for a rank-based combination scheme that was tried and reverted -
        see order_by_similarity_multi's docstring for why).

        Returns [] if candidate_embeddings or reference_embeddings is
        empty.
        """
        if not candidate_embeddings or not reference_embeddings:
            return []
        scores = []
        for ref_emb in reference_embeddings:
            best = None
            for cand_emb in candidate_embeddings:
                score = float(
                    self.recognizer.match(
                        cand_emb, ref_emb, cv2.FaceRecognizerSF_FR_COSINE
                    )
                )
                if best is None or score > best:
                    best = score
            scores.append(best)
        return scores

    def cluster_by_identity(self, paths, threshold=None):
        """Group paths into identity clusters by union-find over pairwise
        cosine similarity. Returns {path: cluster_id}. Images with no
        detected face each get their own singleton cluster.

        Cluster ids are assigned in first-seen order among `paths`, so the
        result is stable/deterministic for a given input order.
        """
        if threshold is None:
            threshold = self.SAME_IDENTITY_THRESHOLD

        embeddings = {p: self.get_embedding(p) for p in paths}
        parent = {p: p for p in paths}

        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        if self.is_available():
            valid = [p for p in paths if embeddings[p] is not None]
            for i, a in enumerate(valid):
                for b in valid[i + 1 :]:
                    if find(a) == find(b):
                        continue
                    score = self.recognizer.match(
                        embeddings[a], embeddings[b], cv2.FaceRecognizerSF_FR_COSINE
                    )
                    if score >= threshold:
                        union(a, b)

        cluster_ids = {}
        result = {}
        for p in paths:
            root = find(p)
            if root not in cluster_ids:
                cluster_ids[root] = len(cluster_ids)
            result[p] = cluster_ids[root]
        return result


_embedder = None


def get_embedder():
    """Get or create the shared FaceEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder()
    return _embedder
