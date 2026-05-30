"""NanoIndex — compressed vector index for any RAG pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .filters import apply_filters
from .quantization.turbo_quant import CompressedDB, TurboQuant

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1


@dataclass
class SearchResult:
    rank:     int
    score:    float
    id:       str
    text:     str
    metadata: dict[str, Any] = field(default_factory=dict)


class NanoIndex:
    """
    Compressed in-memory vector index.

    Compresses float32 embeddings to ~3-4 bits using PolarQuant + QJL,
    achieving 7-10× storage reduction with minimal recall loss.

    Quick start
    -----------
    idx = NanoIndex(dim=384, bits=4)
    idx.add(embeddings, [{"id": "doc1", "text": "...", ...}, ...])
    idx.save("my_index")

    idx = NanoIndex.load("my_index")
    results = idx.search(query_vec, k=10)
    results = idx.search(query_matrix, k=10)   # batch
    """

    def __init__(self, dim: int = 384, bits: int = 4, qjl_m: int = 64, seed: int = 42):
        self.dim   = dim
        self.bits  = bits
        self.qjl_m = qjl_m
        self.seed  = seed
        self.tq    = TurboQuant(dim=dim, bits=bits, qjl_m=qjl_m, seed=seed)
        self._db:   CompressedDB | None = None
        self._meta: list[dict]  = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Compress and store embeddings alongside metadata.

        Parameters
        ----------
        embeddings : (N, dim) float32 — should be L2-normalised
        metadata   : list of N dicts; each dict should contain at least
                     "id" (str) and "text" (str) for search results
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) must match"
            )

        logger.info(
            "Compressing %d vectors (bits=%d, qjl_m=%d)…", len(embeddings), self.bits, self.qjl_m
        )
        self._db   = self.tq.compress(embeddings)
        self._meta = [dict(m) for m in metadata]
        logger.info("Done. Compressed size: %.1f MB", self.tq.memory_mb(len(embeddings)))

        # Warm up Numba JIT so the first real query isn't penalised by compilation
        _dummy = np.zeros(self.dim, dtype=np.float32)
        self.tq.inner_product_batch(_dummy, self._db)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """
        Approximate nearest-neighbour search.

        Parameters
        ----------
        query   : (dim,) or (M, dim) float32 — single or batch query
        k       : number of results per query
        filters : optional metadata filter dict (see filters.apply_filters)

        Returns
        -------
        Single query  → list[SearchResult]
        Batch query   → list[list[SearchResult]]
        """
        if self._db is None:
            raise RuntimeError("Index is empty. Call add() first.")

        query = np.asarray(query, dtype=np.float32)
        batch = query.ndim == 2
        queries = query if batch else query[np.newaxis]

        mask = apply_filters(self._meta, filters)

        all_results = []
        for q in queries:
            all_results.append(self._search_one(q, k, mask))

        return all_results if batch else all_results[0]

    def _search_one(self, q: np.ndarray, k: int, mask: np.ndarray | None) -> list[SearchResult]:
        scores = self.tq.inner_product_batch(q, self._db)

        if mask is not None:
            scores = scores.copy()
            scores[~mask] = -np.inf

        n_valid = int(mask.sum()) if mask is not None else len(scores)
        k_actual = min(k, n_valid)
        if k_actual == 0:
            return []

        top_idx = np.argpartition(scores, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for rank, i in enumerate(top_idx):
            m = self._meta[i]
            results.append(SearchResult(
                rank=rank,
                score=float(scores[i]),
                id=m.get("id", str(i)),
                text=m.get("text", ""),
                metadata={k: v for k, v in m.items() if k not in ("id", "text")},
            ))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to {path}.npz + {path}.meta.json."""
        if self._db is None:
            raise RuntimeError("Index is empty — nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(path),
            angles_int=self._db.angles_int,
            radii=self._db.radii,
            sign_bits=self._db.sign_bits,
            residual_norms=self._db.residual_norms,
            tq_params=np.array([
                _FORMAT_VERSION,
                self._db.n_vectors, self.dim, self.bits, self.qjl_m, self.seed,
            ]),
        )

        meta_path = path.parent / (path.name + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, default=str)

        logger.info("Saved → %s.npz + %s.meta.json", path, path.name)

    @classmethod
    def load(cls, path: str | Path) -> "NanoIndex":
        """Load from {path}.npz + {path}.meta.json. The .npz extension is optional."""
        path = Path(path)
        npz_path  = path if path.suffix == ".npz" else path.parent / (path.name + ".npz")
        meta_path = path.parent / (path.stem + ".meta.json")

        data   = np.load(npz_path)
        params = data["tq_params"].tolist()

        if len(params) == 6:
            _version, n_vec, dim, bits, qjl_m, seed = [int(p) for p in params]
        else:
            # legacy format (no version field)
            n_vec, dim, bits, qjl_m, seed = [int(p) for p in params]

        idx = cls(dim=dim, bits=bits, qjl_m=qjl_m, seed=seed)
        idx._db = CompressedDB(
            angles_int=data["angles_int"],
            radii=data["radii"],
            sign_bits=data["sign_bits"],
            residual_norms=data["residual_norms"],
            n_vectors=n_vec,
            dim=dim,
            bits=bits,
            qjl_m=qjl_m,
        )

        with open(meta_path) as f:
            idx._meta = json.load(f)

        logger.info("Loaded: %d vectors, dim=%d, bits=%d", n_vec, dim, bits)
        return idx

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def n_vectors(self) -> int:
        return self._db.n_vectors if self._db else 0

    def stats(self) -> dict:
        if not self._meta:
            return {}
        return {
            "n_vectors":         self.n_vectors,
            "dim":               self.dim,
            "bits":              self.bits,
            "qjl_m":             self.qjl_m,
            "memory_mb":         round(self.tq.memory_mb(self.n_vectors), 2),
            "compression_ratio": round(self.tq.compression_ratio(), 1),
        }
