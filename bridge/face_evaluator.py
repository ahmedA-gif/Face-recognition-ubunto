"""
Face Model Evaluator — Benchmarks buffalo_s, antelopev2, AdaFace on CCTV data.

Evaluates:
  - FAR (False Acceptance Rate) — wrong person matched
  - FRR (False Rejection Rate) — known person rejected
  - Rank-1 accuracy
  - AUC (Area Under Curve)
  - Inference FPS on CCTV frames

Auto-selects the best model based on evaluation results.
"""

import os
import time
import json
import threading
import numpy as np
import cv2
import requests
from datetime import datetime

FRIGATE_API = os.environ.get("FRIGATE_API", "http://frigate:5000")
EVAL_CACHE_DIR = "/app/data/eval_cache"
os.makedirs(EVAL_CACHE_DIR, exist_ok=True)

MODELS = ["buffalo_s"]  # Start with buffalo_s only; add others after evaluation confirms they work

# Evaluation thresholds to sweep
THRESHOLDS = np.arange(0.2, 0.9, 0.01)


class FaceModelEvaluator:
    """Evaluate and compare InsightFace face recognition models on CCTV data."""

    def __init__(self):
        self._results = {}
        self._best_model = None
        self._lock = threading.Lock()
        self._evaluating = False

    def _load_model(self, model_name):
        """Load InsightFace model with given name."""
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name=model_name,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=0, det_size=(640, 480))
            return app
        except Exception as e:
            print(f"[Evaluator] Failed to load {model_name}: {e}")
            # Try deleting corrupted model and retry
            if "decompress" in str(e) or "block type" in str(e):
                import shutil
                model_dir = os.path.expanduser(f"~/.insightface/models/{model_name}")
                if os.path.exists(model_dir):
                    print(f"[Evaluator] Deleting corrupted model: {model_dir}")
                    shutil.rmtree(model_dir, ignore_errors=True)
                try:
                    app = FaceAnalysis(
                        name=model_name,
                        providers=["CPUExecutionProvider"],
                        allowed_modules=["detection", "recognition"],
                    )
                    app.prepare(ctx_id=0, det_size=(640, 480))
                    return app
                except Exception as e2:
                    print(f"[Evaluator] Retry also failed: {e2}")
            return None

    def _fetch_gallery_images(self):
        """Download face images from Frigate for evaluation."""
        try:
            r = requests.get(f"{FRIGATE_API}/api/faces", timeout=10)
            if r.status_code != 200:
                return {}
            return r.json()
        except Exception:
            return {}

    def _download_image(self, url):
        """Download image from URL."""
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass
        return None

    def _build_pairs(self, faces_data):
        """
        Build genuine and imposter pairs for evaluation.

        Genuine pairs: two different images of the same person
        Imposter pairs: images of different people
        """
        genuine_pairs = []
        imposter_pairs = []

        person_images = {}
        for name, files in faces_data.items():
            if name == "train" or not isinstance(files, list):
                continue
            imgs = []
            for fname in files:
                if not fname.endswith(".webp"):
                    continue
                url = f"{FRIGATE_API}/clips/faces/{name}/{fname}"
                img = self._download_image(url)
                if img is not None:
                    imgs.append(img)
            if imgs:
                person_images[name] = imgs

        names = list(person_images.keys())

        # Genuine pairs: different images of same person
        for name, imgs in person_images.items():
            for i in range(len(imgs)):
                for j in range(i + 1, min(i + 5, len(imgs))):
                    genuine_pairs.append((imgs[i], imgs[j], name, name))

        # Imposter pairs: different people
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                if person_images[n1] and person_images[n2]:
                    # Balance: same number of imposter pairs as genuine pairs for n1
                    genuine_count = len([p for p in genuine_pairs if p[2] == n1])
                    imposter_count = min(3, genuine_count)
                    for k in range(imposter_count):
                        idx1 = min(k, len(person_images[n1]) - 1)
                        idx2 = min(k, len(person_images[n2]) - 1)
                        imposter_pairs.append((
                            person_images[n1][idx1],
                            person_images[n2][idx2],
                            n1, n2
                        ))

        return genuine_pairs, imposter_pairs

    def _compute_embeddings(self, app, images):
        """Compute ArcFace embeddings for a list of images."""
        embeddings = []
        valid_indices = []
        for i, img in enumerate(images):
            try:
                faces = app.get(img)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    emb = best.normed_embedding if hasattr(best, "normed_embedding") else None
                    if emb is None:
                        emb = best.embedding
                        if emb is not None:
                            emb = emb / (np.linalg.norm(emb) + 1e-8)
                    if emb is not None and len(emb) == 512:
                        embeddings.append(emb.astype(np.float32))
                        valid_indices.append(i)
            except Exception:
                continue
        return embeddings, valid_indices

    def _compute_ap(self, fpr, tpr):
        """Compute Area Under Curve using trapezoidal rule."""
        # Sort by FPR
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]
        # Trapezoidal integration
        return np.trapz(tpr_sorted, fpr_sorted)

    def evaluate(self, model_name=None):
        """
        Run full evaluation on all models (or specific model).

        Returns dict with metrics for each model.
        """
        if self._evaluating:
            return {"status": "already_evaluating"}

        self._evaluating = True
        results = {}

        try:
            print("[Evaluator] Fetching gallery images from Frigate...")
            faces_data = self._fetch_gallery_images()
            if not faces_data:
                return {"error": "No face gallery found in Frigate"}

            print("[Evaluator] Building genuine/imposter pairs...")
            genuine_pairs, imposter_pairs = self._build_pairs(faces_data)
            print(f"[Evaluator] {len(genuine_pairs)} genuine pairs, {len(imposter_pairs)} imposter pairs")

            if len(genuine_pairs) < 2 or len(imposter_pairs) < 2:
                return {"error": f"Not enough pairs: {len(genuine_pairs)} genuine, {len(imposter_pairs)} imposter"}

            models_to_eval = [model_name] if model_name else MODELS

            for mname in models_to_eval:
                print(f"\n[Evaluator] === Evaluating {mname} ===")
                app = self._load_model(mname)
                if app is None:
                    results[mname] = {"error": f"Failed to load {mname}"}
                    continue

                start_time = time.time()
                genuine_scores = []
                imposter_scores = []

                # Compute genuine scores
                for img1, img2, n1, n2 in genuine_pairs:
                    faces1 = app.get(img1)
                    faces2 = app.get(img2)
                    if faces1 and faces2:
                        emb1 = faces1[0].normed_embedding if hasattr(faces1[0], "normed_embedding") else faces1[0].embedding
                        emb2 = faces2[0].normed_embedding if hasattr(faces2[0], "normed_embedding") else faces2[0].embedding
                        if emb1 is not None and emb2 is not None:
                            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
                            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
                            score = float(np.dot(emb1, emb2))
                            genuine_scores.append(score)

                # Compute imposter scores
                for img1, img2, n1, n2 in imposter_pairs:
                    faces1 = app.get(img1)
                    faces2 = app.get(img2)
                    if faces1 and faces2:
                        emb1 = faces1[0].normed_embedding if hasattr(faces1[0], "normed_embedding") else faces1[0].embedding
                        emb2 = faces2[0].normed_embedding if hasattr(faces2[0], "normed_embedding") else faces2[0].embedding
                        if emb1 is not None and emb2 is not None:
                            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
                            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
                            score = float(np.dot(emb1, emb2))
                            imposter_scores.append(score)

                elapsed = time.time() - start_time
                fps = (len(genuine_pairs) + len(imposter_pairs)) / elapsed if elapsed > 0 else 0

                # Compute metrics at each threshold
                genuine_scores = np.array(genuine_scores)
                imposter_scores = np.array(imposter_scores)

                fprs = []
                tprs = []
                frrs = []

                for thresh in THRESHOLDS:
                    tp = np.sum(genuine_scores >= thresh)
                    fn = np.sum(genuine_scores < thresh)
                    fp = np.sum(imposter_scores >= thresh)
                    tn = np.sum(imposter_scores < thresh)

                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    frr = fn / (tp + fn) if (tp + fn) > 0 else 0

                    fprs.append(fpr)
                    tprs.append(tpr)
                    frrs.append(frr)

                fprs = np.array(fprs)
                tprs = np.array(tprs)
                frrs = np.array(frrs)

                # Find optimal threshold (min FRR + FPR)
                far_plus_frr = fprs + frrs
                opt_idx = np.argmin(far_plus_frr)
                opt_threshold = THRESHOLDS[opt_idx]

                # AUC
                auc = self._compute_ap(fprs, tprs)

                # Rank-1 accuracy at optimal threshold
                rank1 = tprs[opt_idx]

                # FAR/FRR at optimal threshold
                far_opt = fprs[opt_idx]
                frr_opt = frrs[opt_idx]

                results[mname] = {
                    "model": mname,
                    "genuine_pairs": len(genuine_pairs),
                    "imposter_pairs": len(imposter_pairs),
                    "genuine_scores_mean": float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0,
                    "imposter_scores_mean": float(np.mean(imposter_scores)) if len(imposter_scores) > 0 else 0,
                    "genuine_scores_std": float(np.std(genuine_scores)) if len(genuine_scores) > 0 else 0,
                    "imposter_scores_std": float(np.std(imposter_scores)) if len(imposter_scores) > 0 else 0,
                    "optimal_threshold": float(opt_threshold),
                    "far": float(far_opt),
                    "frr": float(frr_opt),
                    "rank1_accuracy": float(rank1),
                    "auc": float(auc),
                    "cctv_fps": float(fps),
                    "total_time_sec": float(elapsed),
                    "timestamp": datetime.now().isoformat(),
                }

                print(f"[Evaluator] {mname}: FAR={far_opt:.4f} FRR={frr_opt:.4f} "
                      f"Rank1={rank1:.4f} AUC={auc:.4f} FPS={fps:.1f}")

                # Free memory
                del app
                import gc
                gc.collect()

            # Auto-select best model
            best = None
            best_score = -1
            for mname, metrics in results.items():
                if "error" in metrics:
                    continue
                # Score: weighted combination of AUC, low FAR, low FRR
                score = metrics["auc"] * 0.4 + (1 - metrics["far"]) * 0.3 + (1 - metrics["frr"]) * 0.3
                if score > best_score:
                    best_score = score
                    best = mname

            with self._lock:
                self._results = results
                self._best_model = best

            # Save results to file
            report_path = os.path.join(EVAL_CACHE_DIR, "eval_report.json")
            with open(report_path, "w") as f:
                json.dump({"results": results, "best_model": best, "best_score": best_score}, f, indent=2)

            print(f"\n[Evaluator] Best model: {best} (score={best_score:.4f})")
            return results

        finally:
            self._evaluating = False

    def get_report(self):
        """Get the latest evaluation report."""
        with self._lock:
            if self._results:
                return {
                    "results": self._results,
                    "best_model": self._best_model,
                    "status": "complete",
                }
        # Try loading from cache
        report_path = os.path.join(EVAL_CACHE_DIR, "eval_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                data = json.load(f)
                data["status"] = "cached"
                return data
        return {"status": "no_results", "results": {}, "best_model": None}

    def get_best_model(self):
        """Get the best model name from evaluation."""
        with self._lock:
            if self._best_model:
                return self._best_model
        report = self.get_report()
        return report.get("best_model", "buffalo_s")


# Singleton
_evaluator = None


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = FaceModelEvaluator()
    return _evaluator
