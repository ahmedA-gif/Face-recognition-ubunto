"""
FaceEngine — InsightFace ArcFace recognition with FAISS index, margin test, temporal voting.

Pipeline:
  CCTV frame → InsightFace det (buffalo_s) → ArcFace embed (512-D) → FAISS search → margin test → temporal vote → ID

Models:
  - Detection: buffalo_s (RetinaFace)
  - Recognition: buffalo_s or antelopev2 (ArcFace)
  - Fallback: antelopev2 if buffalo_s fails

Key features:
  - FAISS cosine index for fast gallery search
  - Margin test: top1 - top2 >= MIN_MARGIN (prevents Haseeb↔Sulman confusion)
  - Temporal voting: require N consecutive same-name matches before confirming
  - Quality gate: skip low-quality face crops
"""

import os
import time
import threading
import json
import numpy as np
import cv2
import requests
import faiss

# ─── Config ──────────────────────────────────────────────────────────────────
FRIGATE_API = os.environ.get("FRIGATE_API", "http://frigate:5000")
GALLERY_DIR = "/app/data/faces_gallery"
EMBEDDING_CACHE_TTL = 300  # rebuild gallery every 5 min

# Thresholds
MATCH_THRESHOLD = 0.08      # cosine similarity threshold — very low for CCTV (tiny, blurry snapshots)
MIN_MARGIN = 0.01           # top1 - top2 across DIFFERENT identities — minimal margin for CCTV quality
MIN_FACE_SIZE = 40          # minimum face crop size in pixels
QUALITY_MIN = 0.2           # minimum detection score for face to be used
TEMPORAL_WINDOW = 5         # require N consecutive same-name matches
TEMPORAL_CONFIDENCE = 1     # confirm on first match (each Frigate event = separate snapshot, not video frames)

# Model selection
MODEL_NAME = os.environ.get("FACE_MODEL", "buffalo_s")  # buffalo_s or antelopev2
AUTO_SELECT = os.environ.get("FACE_AUTO_SELECT", "true").lower() == "true"


class FaceEngine:
    """InsightFace ArcFace face recognition with FAISS, margin test, temporal voting."""

    def __init__(self):
        self._app = None
        self._gallery_index = None
        self._gallery_names = []
        self._gallery_embeddings = []
        self._lock = threading.Lock()
        self._last_build = 0
        self._temporal_buffer = {}  # track_id -> list of (name, score, timestamp)
        self._built = False

    def _init_model(self):
        """Lazy-init InsightFace model with auto-selection support."""
        if self._app is not None:
            return True

        # Try auto-select best model from evaluation
        selected_model = MODEL_NAME
        if AUTO_SELECT:
            try:
                from face_evaluator import get_evaluator
                evaluator = get_evaluator()
                best = evaluator.get_best_model()
                if best:
                    selected_model = best
                    print(f"[FaceEngine] Auto-selected model: {selected_model}")
            except Exception as e:
                print(f"[FaceEngine] Auto-select failed, using default: {e}")

        # Try loading model, if corrupted, delete and retry
        for attempt in range(2):
            try:
                import insightface
                from insightface.app import FaceAnalysis

                print(f"[FaceEngine] Loading InsightFace model: {selected_model}")
                self._app = FaceAnalysis(
                    name=selected_model,
                    providers=["CPUExecutionProvider"],
                    allowed_modules=["detection", "recognition"],
                )
                self._app.prepare(ctx_id=0, det_size=(640, 480))
                self._model_name = selected_model
                print(f"[FaceEngine] Model loaded: {selected_model}")
                return True
            except Exception as e:
                print(f"[FaceEngine] Failed to load model (attempt {attempt+1}): {e}")
                if "decompress" in str(e) or "block type" in str(e):
                    # Corrupted model, delete and retry
                    import shutil
                    model_dir = os.path.expanduser(f"~/.insightface/models/{selected_model}")
                    if os.path.exists(model_dir):
                        print(f"[FaceEngine] Deleting corrupted model: {model_dir}")
                        shutil.rmtree(model_dir, ignore_errors=True)
                self._app = None
                if attempt == 0:
                    continue
                return False
        return False

    def _build_gallery(self):
        """Build FAISS index from Frigate face directories."""
        if not self._init_model():
            return False

        try:
            r = requests.get(f"{FRIGATE_API}/api/faces", timeout=10)
            if r.status_code != 200:
                print(f"[FaceEngine] Failed to fetch faces: {r.status_code}")
                return False
            faces_data = r.json()
        except Exception as e:
            print(f"[FaceEngine] Error fetching faces: {e}")
            return False

        embeddings = []
        names = []

        for name, files in faces_data.items():
            if name == "train" or not isinstance(files, list):
                continue

            # Use the most recent image for each person
            for fname in sorted(files, reverse=True)[:3]:  # top 3 images per person
                if not fname.endswith(".webp"):
                    continue
                url = f"{FRIGATE_API}/clips/faces/{name}/{fname}"
                try:
                    img_r = requests.get(url, timeout=5)
                    if img_r.status_code != 200:
                        continue
                    arr = np.frombuffer(img_r.content, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue

                    # Detect faces
                    faces = self._app.get(img)
                    if not faces:
                        continue

                    # Use the largest face
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

                    # Quality check
                    det_score = float(best_face.det_score) if hasattr(best_face, "det_score") else 0
                    if det_score < QUALITY_MIN:
                        continue

                    # Get embedding
                    emb = best_face.normed_embedding if hasattr(best_face, "normed_embedding") else None
                    if emb is None:
                        emb = best_face.embedding
                        if emb is not None:
                            emb = emb / (np.linalg.norm(emb) + 1e-8)

                    if emb is not None and len(emb) == 512:
                        embeddings.append(emb.astype(np.float32))
                        names.append(name)
                        print(f"[FaceEngine] Gallery: {name} from {fname} (det={det_score:.3f})")

                except Exception as e:
                    continue

        if not embeddings:
            print("[FaceEngine] No valid embeddings found in gallery")
            return False

        # Build FAISS index (cosine similarity via inner product on normalized vectors)
        dim = len(embeddings[0])
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vectors

        # Normalize all embeddings
        emb_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_array)
        index.add(emb_array)

        with self._lock:
            self._gallery_index = index
            self._gallery_names = names
            self._gallery_embeddings = emb_array
            self._last_build = time.time()
            self._built = True

        print(f"[FaceEngine] Gallery built: {len(names)} embeddings, {len(set(names))} people")
        return True

    def _ensure_gallery(self):
        """Rebuild gallery if stale."""
        if time.time() - self._last_build > EMBEDDING_CACHE_TTL or not self._built:
            self._build_gallery()

    def recognize(self, frame, track_id=None):
        """
        Recognize faces in a frame using InsightFace ArcFace.

        Returns: (name, confidence, top1_score, top2_score, margin) or (None, 0, 0, 0, 0)
        """
        self._ensure_gallery()

        if not self._built or self._gallery_index is None or self._gallery_index.ntotal == 0:
            return None, 0, 0, 0, 0

        if not self._init_model():
            return None, 0, 0, 0, 0

        try:
            faces = self._app.get(frame)
        except Exception as e:
            print(f"[FaceEngine] Detection error: {e}")
            return None, 0, 0, 0, 0

        if not faces:
            print(f"[FaceEngine] No face detected in frame (shape={frame.shape})")
            return None, 0, 0, 0, 0

        # Use the largest/best face
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Quality gate
        det_score = float(best_face.det_score) if hasattr(best_face, "det_score") else 0
        if det_score < QUALITY_MIN:
            print(f"[FaceEngine] Face quality too low: {det_score:.3f} < {QUALITY_MIN}")
            return None, 0, 0, 0, 0
            return None, 0, 0, 0, 0

        # Get embedding
        emb = best_face.normed_embedding if hasattr(best_face, "normed_embedding") else None
        if emb is None:
            emb = best_face.embedding
            if emb is not None:
                emb = emb / (np.linalg.norm(emb) + 1e-8)

        if emb is None or len(emb) != 512:
            return None, 0, 0, 0, 0

        emb_query = emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(emb_query)

        # Search FAISS index — get enough to find 2 different identities
        k = min(self._gallery_index.ntotal, 50)
        scores, indices = self._gallery_index.search(emb_query, k)

        # Find top-2 DIFFERENT identities for margin test
        top1_name = None
        top1_score = 0.0
        top2_name = None
        top2_score = 0.0

        for i in range(min(k, scores.shape[1])):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            if idx < 0:
                continue
            name = self._gallery_names[idx]
            if top1_name is None:
                top1_name = name
                top1_score = score
            elif name != top1_name:
                top2_name = name
                top2_score = score
                break

        margin = top1_score - top2_score if top1_name and top2_name else top1_score

        # ─── Threshold + Margin Test ───
        # Require: top1 above threshold AND margin is large enough
        if top1_score >= MATCH_THRESHOLD and margin >= MIN_MARGIN:
            name = top1_name
            confidence = top1_score
            print(f"[FaceEngine] MATCH: {name} (score={top1_score:.3f}, "
                  f"top2={top2_name}:{top2_score:.3f}, margin={margin:.3f})")
        else:
            name = None
            confidence = 0
            print(f"[FaceEngine] AMBIGUOUS: top1={top1_name}:{top1_score:.3f} "
                  f"top2={top2_name}:{top2_score:.3f} margin={margin:.3f} — UNKNOWN")

        # ─── Temporal Voting ───
        if track_id and name:
            name = self._temporal_vote(track_id, name, confidence)

        return name, confidence, top1_score, top2_score, margin

    def _temporal_vote(self, track_id, name, score):
        """
        Temporal voting: require N consecutive same-name matches before confirming.
        Prevents flickering identity from single-frame misclassifications.
        """
        with self._lock:
            if track_id not in self._temporal_buffer:
                self._temporal_buffer[track_id] = []

            buf = self._temporal_buffer[track_id]
            buf.append((name, score, time.time()))

            # Keep only last N frames
            if len(buf) > TEMPORAL_WINDOW:
                buf.pop(0)

            # Count votes for each name
            votes = {}
            for n, s, t in buf:
                votes[n] = votes.get(n, 0) + 1

            # Find best name
            best_name = max(votes, key=votes.get)
            best_count = votes[best_name]

            if best_count >= TEMPORAL_CONFIDENCE:
                return best_name
            else:
                return None  # Not enough votes yet

    def recognize_from_url(self, image_url, track_id=None):
        """Download image from URL and recognize faces."""
        try:
            r = requests.get(image_url, timeout=5)
            if r.status_code != 200:
                print(f"[FaceEngine] URL fetch failed: {r.status_code} for {image_url}")
                return None, 0, 0, 0, 0
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[FaceEngine] Image decode failed for {image_url}")
                return None, 0, 0, 0, 0
            return self.recognize(img, track_id)
        except Exception as e:
            print(f"[FaceEngine] URL recognition error: {e}")
            return None, 0, 0, 0, 0

    def cleanup_temporal(self, max_age=120):
        """Remove stale temporal buffers."""
        with self._lock:
            now = time.time()
            stale = [k for k, v in self._temporal_buffer.items()
                     if now - v[-1][2] > max_age]
            for k in stale:
                del self._temporal_buffer[k]

    def status(self):
        """Return engine status."""
        return {
            "model": MODEL_NAME,
            "gallery_size": len(self._gallery_names) if self._built else 0,
            "gallery_people": list(set(self._gallery_names)) if self._built else [],
            "index_size": self._gallery_index.ntotal if self._gallery_index else 0,
            "match_threshold": MATCH_THRESHOLD,
            "min_margin": MIN_MARGIN,
            "temporal_window": TEMPORAL_WINDOW,
            "last_build": time.strftime("%H:%M:%S", time.localtime(self._last_build)) if self._last_build else "never",
        }


# ─── Singleton ───────────────────────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
