from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── FAISS-CPU explicit import ──────────────────────────────────────────────────
# Install with:  pip install faiss-cpu
# DO NOT install faiss-gpu on CPU-only machines – it will crash at import time.
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_OK = False
except Exception as exc:  # wrong build (gpu on cpu-only, etc.)
    faiss = None  # type: ignore
    _FAISS_OK = False
    import warnings
    warnings.warn(
        f"[FaceGallery] faiss import failed ({exc}). Falling back to NumPy dot-product search.",
        stacklevel=2,
    )
# ──────────────────────────────────────────────────────────────────────────────


class FaceGallery:
    """SQLite face store with FAISS-CPU cosine (IndexFlatIP) or NumPy search.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    match_threshold : float
        Cosine-similarity cut-off (0–1).  Identities below this are "Unknown".
    backend : {"faiss", "numpy"}
        ``"faiss"``  – use faiss-cpu IndexFlatIP (fast, recommended).
        ``"numpy"``  – pure-NumPy dot-product (always available, no extra dep).
        If ``"faiss"`` is requested but the library is not installed the code
        silently degrades to ``"numpy"``.
    index_path : str | None
        Where to persist the FAISS index between runs.  Defaults to
        ``<db_path>.faiss`` next to the SQLite DB.
    """

    def __init__(
        self,
        db_path: str,
        match_threshold: float = 0.42,
        backend: str = "faiss",
        index_path: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.match_threshold = match_threshold
        self.index_path = index_path or str(Path(db_path).with_suffix(".faiss"))

        # Resolve effective backend
        _req = backend.lower().strip()
        if _req == "faiss" and not _FAISS_OK:
            import warnings
            warnings.warn(
                "[FaceGallery] faiss-cpu not installed. "
                "Run: pip install faiss-cpu  — falling back to NumPy.",
                stacklevel=2,
            )
            self.backend = "numpy"
        else:
            self.backend = _req

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT    NOT NULL,
                embedding BLOB    NOT NULL,
                dim       INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)"
        )
        self._conn.commit()

        self._names: List[str] = []
        self._matrix: Optional[np.ndarray] = None   # (capacity, dim) float32 for numpy path
        self._n = 0                                 # used rows in _matrix
        self._faiss_index = None                     # faiss.IndexFlatIP
        self._dirty_persist = False

        self.reload()

    # ── helpers ───────────────────────────────────────────────────────────────

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

    # ── FAISS index builder ────────────────────────────────────────────────────

    def _load_rows(self) -> Tuple[List[str], Optional[np.ndarray]]:
        """Stream rows from SQLite in chunks (memory-friendly for big galleries).

        Returns ``(names, matrix)`` where matrix rows are L2-normalised.
        Corrupt rows (embedding size != declared dim) are skipped.
        """
        names: List[str] = []
        chunks: List[np.ndarray] = []
        cur = self._conn.execute(
            "SELECT name, embedding, dim FROM faces ORDER BY id ASC"
        )
        while True:
            batch = cur.fetchmany(512)
            if not batch:
                break
            for row in batch:
                name, blob, dim = row["name"], row["embedding"], row["dim"]
                emb = np.frombuffer(blob, dtype=np.float32)
                if emb.size != dim:
                    continue
                names.append(name)
                chunks.append(self._normalize(emb))
        matrix = np.vstack(chunks) if chunks else None
        return names, matrix

    def _load_persisted_index(self, expected: int, dim: int) -> object | None:
        """Try to reuse the on-disk FAISS index.

        Only used when it exactly matches the table (row count + dimension);
        otherwise ``None`` signals a rebuild from SQLite.
        """
        if self.backend != "faiss" or faiss is None:
            return None
        path = Path(self.index_path)
        if not path.exists():
            return None
        try:
            index = faiss.read_index(self.index_path)
            if index.ntotal != expected or index.d == dim:
                return None
            return index
        except Exception:
            return None

    def _build_faiss_index(self, vecs: List[np.ndarray], *, persist: bool = True) -> None:
        """Build (or rebuild) the in-memory FAISS IndexFlatIP from normalised vectors."""
        self._faiss_index = None
        if self.backend != "faiss" or faiss is None or not vecs:
            return

        matrix = np.stack(vecs, axis=0).astype(np.float32)
        dim = int(matrix.shape[1])

        # IndexFlatIP = exact inner-product search on unit vectors == cosine
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        self._faiss_index = index

        if persist:
            self._persist_faiss()
        else:
            self._dirty_persist = True

    def _matrix_view(self) -> Optional[np.ndarray]:
        """The used portion of the capacity matrix (numpy fallback path)."""
        if self._matrix is None:
            return None
        return self._matrix[: self._n]

    def _append_in_memory(self, name: str, emb: np.ndarray) -> None:
        """O(1) amortised append into search structures without full DB reload."""
        self._names.append(name)
        row = emb.reshape(1, -1).astype(np.float32)
        dim = row.shape[1]

        if self._matrix is None:
            self._matrix = np.empty((64, dim), np.float32)
            self._n = 0
        if self._n >= self._matrix.shape[0]:
            # capacity doubling — avoids O(N²) vstack on every add
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

    # ── public API ────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Re-read the SQLite table and (re)build the search index.

        - Rows are streamed in chunks (not one big fetchall).
        - The persisted FAISS index is reused when it exactly matches the
          table (count + dim); otherwise it is rebuilt from SQLite.
        """
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
        """Insert one embedding for *name*.

        By default appends in-memory (O(1)); set ``rebuild=True`` to force a
        full reload from SQLite (useful after bulk external writes).
        """
        emb = self._normalize(embedding)
        self._conn.execute(
            "INSERT INTO faces(name, embedding, dim) VALUES (?, ?, ?)",
            (name, emb.tobytes(), int(emb.size)),
        )
        self._conn.commit()
        if rebuild:
            self.reload()
        else:
            self._append_in_memory(name, emb)

    def add_many(self, items: List[Tuple[str, np.ndarray]]) -> int:
        """Bulk-insert embeddings with a single commit + single index rebuild."""
        if not items:
            return 0
        rows = []
        for name, embedding in items:
            emb = self._normalize(embedding)
            rows.append((name, emb.tobytes(), int(emb.size)))
        self._conn.executemany(
            "INSERT INTO faces(name, embedding, dim) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self.reload()
        return len(rows)

    def match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Return (name, cosine_score). Returns ("Unknown", score) if below threshold."""
        if self._matrix is None or len(self._names) == 0:
            return "Unknown", 0.0

        emb = self._normalize(embedding)

        if self.backend == "faiss" and self._faiss_index is not None:
            # FAISS-CPU path — unit vectors ⟹ inner product == cosine similarity
            scores, indices = self._faiss_index.search(emb.reshape(1, -1), 1)
            idx = int(indices[0][0])
            score = float(scores[0][0])
            # FAISS returns -1 when the index is empty / no neighbour found
            if idx < 0 or idx >= len(self._names):
                return "Unknown", 0.0
        else:
            # NumPy fallback — dot product of unit vectors == cosine
            sims = self._matrix_view() @ emb
            idx = int(np.argmax(sims))
            score = float(sims[idx])

        if score >= self.match_threshold:
            return self._names[idx], score
        return "Unknown", score

    def list_people(self) -> List[str]:
        """Return sorted unique names currently in the gallery (from memory)."""
        return sorted(set(self._names))

    def count(self) -> int:
        """Total number of embeddings currently loaded in the search index."""
        return len(self._names)

    def count_db(self) -> int:
        """Authoritative row count from SQLite (hits disk)."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()
        return int(row["n"])

    def enroll_folder(self, images_dir: str, face_engine) -> Dict[str, int]:
        """Bulk-enroll from  ``<images_dir>/<PersonName>/*.jpg``.

        Example layout::

            data/faces_gallery/
                Ahmed/   photo1.jpg  photo2.png
                Sara/    headshot.jpg

        Only the *largest* detected face per image is enrolled.
        Embeddings are batched then committed once for speed.
        """
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
                # Pick the largest detected face (most likely the subject)
                hit = max(hits, key=lambda h: (h.xyxy[2] - h.xyxy[0]) * (h.xyxy[3] - h.xyxy[1]))
                batch.append((name, hit.embedding))
                n += 1
            counts[name] = n

        if batch:
            self.add_many(batch)
        return counts

    def flush(self) -> None:
        """Persist a dirty FAISS index to disk if needed."""
        if self._dirty_persist:
            self._persist_faiss()

    def close(self) -> None:
        self.flush()
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def status(self) -> str:
        """One-line status string for logging / HUD display (memory-backed, no extra queries)."""
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
