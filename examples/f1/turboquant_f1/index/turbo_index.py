"""TurboQuantIndex — build, search, save, and load a compressed vector index."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..quantization.turbo_quant import CompressedDB, TurboQuant
from .filters import apply_filters

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    rank:            int
    score:           float
    id:              str
    text:            str
    chunk_type:      str
    year:            int
    race:            str
    session:         str
    driver:          str
    lap:             int
    timestamp_start: float
    timestamp_end:   float
    compound:        Optional[str]
    position:        Optional[int]


class TurboQuantIndex:
    """
    In-memory compressed vector index.

    Typical usage
    -------------
    idx = TurboQuantIndex(dim=384, bits=3)
    idx.add(embeddings, chunks)      # chunks: list[dict] from chunks.json
    idx.save("data/index/bahrain_r")

    idx2 = TurboQuantIndex.load("data/index/bahrain_r")
    results = idx2.search(query_vec, k=10, filters={"driver": "VER"})
    """

    def __init__(self, dim: int = 384, bits: int = 3, qjl_m: int = 64, seed: int = 42):
        self.dim    = dim
        self.bits   = bits
        self.qjl_m  = qjl_m
        self.seed   = seed
        self.tq     = TurboQuant(dim=dim, bits=bits, qjl_m=qjl_m, seed=seed)
        self._db: Optional[CompressedDB] = None
        self._meta: list[dict] = []   # one dict per vector (chunk fields)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Compress embeddings and store alongside chunk metadata.

        Parameters
        ----------
        embeddings : (N, dim) float32
        chunks     : list of N chunk dicts (from chunks.json)
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"embeddings ({len(embeddings)}) and chunks ({len(chunks)}) must match")

        logger.info("Compressing %d vectors (bits=%d, qjl_m=%d)…", len(embeddings), self.bits, self.qjl_m)
        self._db   = self.tq.compress(np.asarray(embeddings, dtype=np.float32))
        self._meta = [dict(c) for c in chunks]
        logger.info("Compression done. Index size: %.1f MB", self.tq.memory_mb(len(embeddings)))

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Approximate nearest-neighbour search.

        Parameters
        ----------
        query   : (dim,) float32  — should be L2-normalised
        k       : int             — number of results
        filters : dict | None     — see filters.apply_filters for supported keys

        Returns
        -------
        list of SearchResult, sorted by descending score
        """
        if self._db is None:
            raise RuntimeError("Index is empty. Call add() or load() first.")

        scores = self.tq.inner_product_batch(query, self._db)  # (N,)

        # Apply metadata filter by setting masked scores to -inf
        mask = apply_filters(self._meta, filters)
        if mask is not None:
            scores[~mask] = -np.inf

        # Top-k retrieval (partial sort)
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
                id=m.get("id", ""),
                text=m.get("text", ""),
                chunk_type=m.get("chunk_type", ""),
                year=m.get("year", 0),
                race=m.get("race", ""),
                session=m.get("session", ""),
                driver=m.get("driver", ""),
                lap=m.get("lap", 0),
                timestamp_start=m.get("timestamp_start", 0.0),
                timestamp_end=m.get("timestamp_end", 0.0),
                compound=m.get("compound"),
                position=m.get("position"),
            ))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save index to {path}.npz (compressed arrays) + {path}.meta.json (chunk metadata).
        """
        if self._db is None:
            raise RuntimeError("Nothing to save — index is empty.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Compressed vectors
        np.savez_compressed(
            str(path),
            angles_int=self._db.angles_int,
            radii=self._db.radii,
            sign_bits=self._db.sign_bits,
            residual_norms=self._db.residual_norms,
            tq_params=np.array([self._db.n_vectors, self.dim, self.bits, self.qjl_m, self.seed]),
        )

        # Chunk metadata
        meta_path = path.parent / (path.name + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, default=str)

        npz_size = (path.parent / (path.name + ".npz")).stat().st_size / 1e6
        logger.info("Saved index → %s.npz (%.1f MB) + %s.meta.json", path, npz_size, path.name)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantIndex":
        """
        Load index from {path}.npz + {path}.meta.json.
        The .npz extension is optional — it will be appended if missing.
        """
        path = Path(path)
        npz_path  = path if path.suffix == ".npz" else path.parent / (path.name + ".npz")
        meta_path = path.parent / (path.stem + ".meta.json")

        data   = np.load(npz_path)
        params = data["tq_params"].tolist()
        n_vec, dim, bits, qjl_m, seed = int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4])

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

        logger.info("Loaded index: %d vectors, dim=%d, bits=%d", n_vec, dim, bits)
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
        from collections import Counter
        return {
            "n_vectors":   self.n_vectors,
            "dim":         self.dim,
            "bits":        self.bits,
            "qjl_m":       self.qjl_m,
            "chunk_types": dict(Counter(m["chunk_type"] for m in self._meta)),
            "races":       sorted(set(m["race"] for m in self._meta)),
            "drivers":     sorted(set(m["driver"] for m in self._meta)),
            "memory_mb":   round(self.tq.memory_mb(self.n_vectors), 1),
            "compression_ratio": round(self.tq.compression_ratio(), 1),
        }
