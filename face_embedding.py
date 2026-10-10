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

    def compute_similarity_to_set(self, path, reference_embeddings):
        """Nearest-neighbor cosine similarity between ANY detected face in
        path and a SET of reference embeddings - the best (highest) match
        across every (path face, reference face) pair, rather than a
        single reference or a single largest-bounding-box face on either
        side.

        A multi-reference gallery is more robust than one reference photo:
        matching against whichever reference is closest means one unusual
        angle, lighting, or expression among the references can't drag an
        otherwise-good match down the way averaging embeddings together
        would, and it still works sensibly if the references are actually
        several different people, where no single "average face" is
        meaningful at all. Matching every face detected in path (not just
        the largest) means a photo where the subject isn't the most
        prominent face in frame still gets compared correctly. Returns
        None if path has no detectable face or no reference embedding is
        available.
        """
        if not self.is_available():
            return None
        candidate_embeddings = self.get_face_embeddings(path)
        if not candidate_embeddings:
            return None
        best = None
        for emb in candidate_embeddings:
            score = self._best_match(emb, reference_embeddings)
            if score is not None and (best is None or score > best):
                best = score
        return best

    def compute_discriminative_similarity(
        self, path, positive_embeddings, negative_embeddings
    ):
        """Nearest-neighbor similarity to a positive reference set, minus
        nearest-neighbor similarity to a negative set - rewards matching
        the positives AND being unlike the negatives, rather than scoring
        purely on how well path matches the positives in isolation.

        This catches a lookalike that compute_similarity_to_set alone
        would rank highly: if they also score high against the negative
        set (e.g. because the negatives happen to share lighting, studio,
        or age bracket with the positives rather than being a clean
        contrast set), that drags their score back down instead of
        leaving them sitting near the top.

        Each face detected in path is scored independently (its own best
        match against the positives minus its own best match against the
        negatives), and the highest-scoring face wins - so in a multi-face
        photo, one face resembling a negative doesn't drag down a
        DIFFERENT face in the same photo that's a genuine positive match.

        Returns None if path has no detectable face or none of the
        positive embeddings are usable. An empty/all-None negative set
        just means no penalty is applied - equivalent to
        compute_similarity_to_set.
        """
        if not self.is_available():
            return None
        candidate_embeddings = self.get_face_embeddings(path)
        if not candidate_embeddings:
            return None

        best_score = None
        for emb in candidate_embeddings:
            best_positive = self._best_match(emb, positive_embeddings)
            if best_positive is None:
                continue
            best_negative = self._best_match(emb, negative_embeddings)
            score = (
                best_positive
                if best_negative is None
                else best_positive - best_negative
            )
            if best_score is None or score > best_score:
                best_score = score
        return best_score

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
