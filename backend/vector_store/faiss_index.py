"""
FAISS Vector Store Manager
============================
Module  : vector_store/faiss_index.py
Purpose : Manage the FAISS flat index for fast nearest-neighbour search
          across all stored signature embeddings.

Author  : Signature Verifier Team
Version : 1.0.0
"""

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from backend.core.exceptions import VectorStoreError
from backend.core.logger import get_logger

log = get_logger("faiss_index")


class FAISSIndexManager:
    """
    Thread-safe wrapper around a FAISS IndexFlatIP (inner-product / cosine) index.

    Because embeddings are L2-normalised by the Siamese encoder, inner-product
    equals cosine similarity — so IndexFlatIP is the correct choice here.

    The index is kept fully in-memory during runtime and persisted to disk on
    every write operation to survive restarts.

    Attributes:
        index_path   : Filesystem path for the persisted .faiss file.
        dim          : Embedding dimension (must match model output).
        _index       : The underlying faiss.IndexFlatIP object.
        _lock        : Re-entrant lock for thread safety.
        _id_map      : Maps FAISS row index → database Signature.id.
    """

    def __init__(self, index_path: str, dim: int = 512) -> None:
        self.index_path = Path(index_path)
        self.dim = dim
        self._lock = threading.RLock()
        self._id_map: List[int] = []   # position i → db signature id
        self._index: Optional[faiss.IndexFlatIP] = None
        self._load_or_create()

    # ─── Public API ───────────────────────────────────────────────────────────

    def add(self, embedding: np.ndarray, db_sig_id: int) -> int:
        """
        Add a single embedding to the index.

        Args:
            embedding  : 1-D float32 array of shape (dim,). Must be L2-normalised.
            db_sig_id  : The database primary key to associate with this vector.

        Returns:
            int: The FAISS row index (position) of the newly added vector.

        Raises:
            VectorStoreError: If the add operation fails.
        """
        try:
            with self._lock:
                vec = self._prepare(embedding)
                self._index.add(vec)
                faiss_id = len(self._id_map)
                self._id_map.append(db_sig_id)
                self._persist()
                log.info(f"Vector added | faiss_id={faiss_id} | db_sig_id={db_sig_id}")
                return faiss_id
        except Exception as exc:
            log.error(f"FAISS add failed | db_sig_id={db_sig_id} | error={exc}")
            raise VectorStoreError("add", detail=str(exc)) from exc

    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find the top-k most similar vectors to the query embedding.

        Args:
            query : 1-D float32 array of shape (dim,). Must be L2-normalised.
            top_k : Number of nearest neighbours to return.

        Returns:
            List of (db_sig_id, cosine_score) tuples sorted by score descending.

        Raises:
            VectorStoreError: If the index is empty or search fails.
        """
        try:
            with self._lock:
                if self._index.ntotal == 0:
                    log.warning("FAISS search on empty index")
                    return []

                actual_k = min(top_k, self._index.ntotal)
                vec = self._prepare(query)
                scores, indices = self._index.search(vec, actual_k)

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    db_id = self._id_map[idx]
                    results.append((db_id, float(score)))

                log.debug(f"FAISS search | top_k={actual_k} | found={len(results)}")
                return results
        except Exception as exc:
            log.error(f"FAISS search failed | error={exc}")
            raise VectorStoreError("search", detail=str(exc)) from exc

    def rebuild(self, embeddings: List[Tuple[int, np.ndarray]]) -> None:
        """
        Rebuild the index from scratch from a list of (db_sig_id, embedding) pairs.

        Used after bulk deletions or when the index drifts out of sync with the DB.

        Args:
            embeddings: List of (db_sig_id, float32_array) pairs.
        """
        try:
            with self._lock:
                self._index = faiss.IndexFlatIP(self.dim)
                self._id_map = []

                if embeddings:
                    matrix = np.vstack([emb for _, emb in embeddings]).astype(np.float32)
                    faiss.normalize_L2(matrix)
                    self._index.add(matrix)
                    self._id_map = [db_id for db_id, _ in embeddings]

                self._persist()
                log.info(f"FAISS index rebuilt | total_vectors={self._index.ntotal}")
        except Exception as exc:
            log.error(f"FAISS rebuild failed | error={exc}")
            raise VectorStoreError("rebuild", detail=str(exc)) from exc

    @property
    def total(self) -> int:
        """Number of vectors currently stored in the index."""
        with self._lock:
            return self._index.ntotal if self._index else 0

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        """Load the persisted index from disk, or create a new empty index."""
        try:
            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                id_map_path = self.index_path.with_suffix(".idmap.npy")
                if id_map_path.exists():
                    self._id_map = list(np.load(str(id_map_path)).astype(int))
                log.info(
                    f"FAISS index loaded | path={self.index_path} | "
                    f"vectors={self._index.ntotal}"
                )
            else:
                self._index = faiss.IndexFlatIP(self.dim)
                log.info(f"New FAISS index created | dim={self.dim}")
        except Exception as exc:
            log.warning(f"Failed to load FAISS index, creating new | error={exc}")
            self._index = faiss.IndexFlatIP(self.dim)
            self._id_map = []

    def _persist(self) -> None:
        """Write the current index and id-map to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            id_map_path = self.index_path.with_suffix(".idmap.npy")
            np.save(str(id_map_path), np.array(self._id_map, dtype=np.int64))
        except Exception as exc:
            log.error(f"FAISS persist failed | error={exc}")

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        """Reshape and L2-normalise a 1-D embedding for FAISS."""
        vec = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec
