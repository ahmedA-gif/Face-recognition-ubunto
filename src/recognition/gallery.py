from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.db.postgres import execute_query, execute_write, execute_insert, execute_many

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    faiss = None
    _FAISS_OK = False
except Exception:
    faiss = None
    _FAISS_OK = False
    import warnings
    warnings.warn(
        f"[FaceGallery] faiss import failed. Falling back to NumPy dot-product search.",
        stacklevel=2,
    )


class FaceGallery:
    """PostgreSQL face store with FAISS-CPU cosine or NumPy search."""

    def __init__(
        self,
        db_path: str = "",
        match_threshold: float = 0.48,
        ambiguity_margin: float = 0.08,
        backend: str = "faiss",
        index_path: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self.index_path = index_path or "data/db/faces.faiss"

        _req = backend.lower().strip()
        if _req == "faiss" and not _FAISS_OK:
            import warnings
            warnings.warn(
                "[FaceGallery] faiss-cpu not installed. Falling back to NumPy.",
                stacklevel=2,
            )
            self.backend = "numpy"
        else:
            self.backend = _req

        self._names: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._n = 0
        self._faiss_index = None
        self._dirty_persist = False

        self.reload()

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        emb = embedding.astype(np.float32).ravel()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _persist_faiss(self) -> None:
        if self._faiss_index is None or faiss is None:
            return
        try:
            faiss.write_index(self._faiss_index, self.index_path)
            self._dirty_persist = False
        except Exception:
            pass

    def _load_rows(self) -> Tuple[List[str], Optional[np.ndarray]]:
        names: List[str] = []
        chunks: List[np.ndarray] = []
        rows = execute_query("SELECT name, embedding, dim FROM faces ORDER BY id ASC")
        for row in rows:
            name = row["name"]
            blob = row["embedding"]
            dim = row["dim"]
            if isinstance(blob, memoryview):
                blob = bytes(blob)
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size != dim:
                continue
            names.append(name)
            chunks.append(self._normalize(emb))
        matrix = np.vstack(chunks) if chunks else None
        return names, matrix

    def _load_persisted_index(self, expected: int, dim: int) -> object | None:
        if self.backend != "faiss" or faiss is None:
            return None
        path = Path(self.index_path)
        if not path.exists():
            return None
        try:
            index = faiss.read_index(self.index_path)
            if index.ntotal != expected or index.d != dim:
                return None
            return index
        except Exception:
            return None

    def _build_faiss_index(self, vecs: List[np.ndarray], *, persist: bool = True) -> None:
        self._faiss_index = None
        if self.backend != "faiss" or faiss is None or not vecs:
            return
        matrix = np.stack(vecs, axis=0).astype(np.float32)
        dim = int(matrix.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        self._faiss_index = index
        if persist:
            self._persist_faiss()
        else:
            self._dirty_persist = True

    def _matrix_view(self) -> Optional[np.ndarray]:
        if self._matrix is None:
            return None
        return self._matrix[: self._n]

    def _append_in_memory(self, name: str, emb: np.ndarray) -> None:
        self._names.append(name)
        row = emb.reshape(1, -1).astype(np.float32)
        dim = row.shape[1]
        if self._matrix is None:
            self._matrix = np.empty((64, dim), np.float32)
            self._n = 0
        if self._n >= self._matrix.shape[0]:
            grown = np.empty((self._matrix.shape[0] * 2, dim), np.float32)
            grown[: self._n] = self._matrix[: self._n]
            self._matrix = grown
        self._matrix[self._n] = row
        self._n += 1
        if self.backend == "faiss" and faiss is not None:
            if self._faiss_index is None:
                self._build_faiss_index([emb], persist=False)
            else:
                self._faiss_index.add(row)
                self._dirty_persist = True

    def reload(self) -> None:
        self._names, self._matrix = self._load_rows()
        self._n = len(self._names)
        if self.backend == "faiss" and faiss is not None and self._n > 0:
            dim = int(self._matrix.shape[1]) if self._matrix is not None else 0
            loaded = self._load_persisted_index(self._n, dim)
            if loaded is not None:
                self._faiss_index = loaded
                self._dirty_persist = False
                return
            self._build_faiss_index(self._names2vecs(), persist=True)
        else:
            self._faiss_index = None

    def _names2vecs(self) -> List[np.ndarray]:
        if self._matrix is None:
            return []
        return [self._matrix[i] for i in range(self._n)]

    def add(self, name: str, embedding: np.ndarray, *, rebuild: bool = False) -> None:
        emb = self._normalize(embedding)
        execute_insert(
            "INSERT INTO faces(name, embedding, dim) VALUES (%s, %s, %s) RETURNING id",
            (name, emb.tobytes(), int(emb.size)),
        )
        if rebuild:
            self.reload()
        else:
            self._append_in_memory(name, emb)

    def add_many(self, items: List[Tuple[str, np.ndarray]]) -> int:
        if not items:
            return 0
        rows = []
        for name, embedding in items:
            emb = self._normalize(embedding)
            rows.append((name, emb.tobytes(), int(emb.size)))
        execute_many("INSERT INTO faces(name, embedding, dim) VALUES (%s, %s, %s)", rows)
        self.reload()
        # Batch enrollment must leave the on-disk FAISS index in step with the
        # committed DB rows, even if the process is interrupted afterward.
        self._persist_faiss()
        return len(rows)

    def match(self, embedding: np.ndarray) -> Tuple[str, float]:
        if self._matrix is None or len(self._names) == 0:
            return "Unknown", 0.0
        emb = self._normalize(embedding)
        second_score = -1.0
        if self.backend == "faiss" and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(emb.reshape(1, -1), min(2, len(self._names)))
            idx = int(indices[0][0])
            score = float(scores[0][0])
            if scores.shape[1] > 1:
                second_score = float(scores[0][1])
            if idx < 0 or idx >= len(self._names):
                return "Unknown", 0.0
        else:
            sims = self._matrix_view() @ emb
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            if len(sims) > 1:
                second_score = float(np.partition(sims, -2)[-2])
        # A close runner-up is not an identity decision. Returning Unknown here
        # prevents an ambiguous face from contaminating fusion/attendance.
        if score >= self.match_threshold and (second_score < 0 or score - second_score >= self.ambiguity_margin):
            return self._names[idx], score
        return "Unknown", score

    def list_people(self) -> List[str]:
        return sorted(set(self._names))

    def count(self) -> int:
        return len(self._names)

    def count_db(self) -> int:
        rows = execute_query("SELECT COUNT(*) AS n FROM faces")
        return int(rows[0]["n"]) if rows else 0

    def enroll_folder(self, images_dir: str, face_engine) -> Dict[str, int]:
        root = Path(images_dir)
        counts: Dict[str, int] = {}
        if not root.exists():
            return counts
        batch: List[Tuple[str, np.ndarray]] = []
        for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            name = person_dir.name
            n = 0
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                hits = face_engine.detect_and_embed(img, min_face_px=20)
                if not hits:
                    continue
                hit = max(hits, key=lambda h: (h.xyxy[2] - h.xyxy[0]) * (h.xyxy[3] - h.xyxy[1]))
                batch.append((name, hit.embedding))
                n += 1
            counts[name] = n
        if batch:
            self.add_many(batch)
        return counts

    def flush(self) -> None:
        if self._dirty_persist:
            self._persist_faiss()

    def close(self) -> None:
        self.flush()

    def status(self) -> str:
        backend_str = "faiss-cpu" if (self.backend == "faiss" and _FAISS_OK) else "numpy"
        return (
            f"FaceGallery | backend={backend_str} | "
            f"people={len(set(self._names))} | embeddings={len(self._names)} | "
            f"threshold={self.match_threshold:.2f}"
        )

    def __enter__(self) -> "FaceGallery":
        return self

    def __exit__(self, *_) -> None:
        self.close()
