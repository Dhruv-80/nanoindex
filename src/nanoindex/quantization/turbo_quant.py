"""
TurboQuant: combined PolarQuant + QJL compression pipeline.

Storage per vector (d=384, bits=3, m=64):
  PolarQuant angles : 383 * 3 = 1149 bits → 144 bytes (packed)
  PolarQuant radius :   1 * 32            =   4 bytes
  QJL sign bits     :  64 * 1             =   8 bytes (packed)
  QJL residual norm :   1 * 32            =   4 bytes
  ─────────────────────────────────────────────────────
  Total             :                       160 bytes
  vs float32 raw    : 384 * 4             = 1536 bytes
  Compression ratio :                       ~9.6×
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .polar_quant import PolarQuant
from .qjl import QJL


@dataclass
class CompressedDB:
    """All compressed data needed for search."""
    angles_int:     np.ndarray  # (N, d-1) int32
    radii:          np.ndarray  # (N,) float32
    sign_bits:      np.ndarray  # (N, m) int8
    residual_norms: np.ndarray  # (N,) float32
    n_vectors:      int
    dim:            int
    bits:           int
    qjl_m:         int


class TurboQuant:
    def __init__(
        self,
        dim: int,
        bits: int = 3,
        qjl_m: int = 64,
        seed: int = 42,
    ):
        self.dim   = dim
        self.bits  = bits
        self.qjl_m = qjl_m

        self.polar = PolarQuant(dim, bits=bits, seed=seed)
        # QJL operates in the rotated space — same seed offset to keep matrices independent
        self.qjl   = QJL(dim, m=qjl_m, seed=seed + 1)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, vectors: np.ndarray) -> CompressedDB:
        """
        Compress a batch of float32 vectors.

        Parameters
        ----------
        vectors : (N, d) float32

        Returns
        -------
        CompressedDB
        """
        vectors = np.asarray(vectors, dtype=np.float32)

        # Step 1: PolarQuant compress
        angles_int, radii = self.polar.compress(vectors)

        # Step 2: compute residual in the rotated space (where QJL will operate)
        v_rot        = vectors @ self.polar.R.T                             # (N, d)
        v_approx_rot = self.polar.reconstruct(angles_int, radii) @ self.polar.R.T  # (N, d)
        residuals    = v_rot - v_approx_rot                                 # (N, d)

        # Step 3: QJL compress the residuals
        sign_bits, residual_norms = self.qjl.compress_residuals(residuals)

        return CompressedDB(
            angles_int=angles_int,
            radii=radii,
            sign_bits=sign_bits,
            residual_norms=residual_norms,
            n_vectors=len(vectors),
            dim=self.dim,
            bits=self.bits,
            qjl_m=self.qjl_m,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def inner_product_batch(
        self,
        query: np.ndarray,
        db: CompressedDB,
    ) -> np.ndarray:
        """
        Approximate inner products between one query and all N compressed vectors.

        Parameters
        ----------
        query : (d,) float32  (should be L2-normalised for cosine similarity)
        db    : CompressedDB

        Returns
        -------
        scores : (N,) float32
        """
        q = np.asarray(query, dtype=np.float32)
        q_rot = self.polar.R @ q  # rotate query (same R)

        # PolarQuant approximate inner product
        polar_scores = self.polar.inner_product_batch(q, db.angles_int, db.radii)

        # QJL correction
        correction = self.qjl.correction_batch(q_rot, db.sign_bits, db.residual_norms)

        return polar_scores + correction

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, db: CompressedDB, path: str) -> None:
        np.savez_compressed(
            path,
            angles_int=db.angles_int,
            radii=db.radii,
            sign_bits=db.sign_bits,
            residual_norms=db.residual_norms,
            meta=np.array([db.n_vectors, db.dim, db.bits, db.qjl_m]),
        )

    def load(self, path: str) -> CompressedDB:
        data = np.load(path)
        meta = data["meta"]
        return CompressedDB(
            angles_int=data["angles_int"],
            radii=data["radii"],
            sign_bits=data["sign_bits"],
            residual_norms=data["residual_norms"],
            n_vectors=int(meta[0]),
            dim=int(meta[1]),
            bits=int(meta[2]),
            qjl_m=int(meta[3]),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compression_ratio(self) -> float:
        float32_bytes = self.dim * 4
        compressed_bytes = self.polar.bytes_per_vector() + self.qjl.bytes_per_vector()
        return float32_bytes / compressed_bytes

    def memory_mb(self, n_vectors: int) -> float:
        compressed_bytes = self.polar.bytes_per_vector() + self.qjl.bytes_per_vector()
        return n_vectors * compressed_bytes / 1e6
