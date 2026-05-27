"""
QJL (quantized Johnson-Lindenstrauss) residual correction.

After PolarQuant compresses a vector v, the quantization residual
  e = v_rotated - PolarQuant.reconstruct(v_rotated)
carries information about the approximation error.

QJL projects e through a random matrix S and stores only the sign bits:
  b = sign(S @ e)   ∈ {-1, +1}^m   (m bits per vector)

At query time, the correction to the inner product estimate is:
  ⟨q', e⟩ ≈ sqrt(π/2) * ‖e‖ * (1/m) * b · (S @ q')

where q' = R @ q is the rotated query (same R as PolarQuant).

This is unbiased under the Gaussian approximation:
  E[sign(S_i · e) · (S_i · q')] = sqrt(2/π) · ⟨q', e⟩ / ‖e‖
so the estimator equals ⟨q', e⟩ in expectation.
"""

from __future__ import annotations

import numpy as np


class QJL:
    def __init__(self, dim: int, m: int = 64, seed: int = 42):
        """
        Parameters
        ----------
        dim : int   dimension of the (rotated) space
        m   : int   number of JL projection dimensions (= bits stored per vector)
        seed: int   random seed
        """
        self.dim = dim
        self.m = m

        rng = np.random.default_rng(seed)
        # Gaussian random matrix; rows are normalised to unit length
        S = rng.standard_normal((m, dim)).astype(np.float32)
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        self.S = S / norms  # (m, dim), unit-norm rows

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_residuals(
        self,
        residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        residuals : (N, d) float32   v_rotated - polar_reconstruct(v_rotated)

        Returns
        -------
        sign_bits      : (N, m) int8  {-1, +1}
        residual_norms : (N,) float32  ‖e‖ per vector (needed for unbiased estimate)
        """
        residuals = np.asarray(residuals, dtype=np.float32)
        single = residuals.ndim == 1
        if single:
            residuals = residuals[np.newaxis]

        z = residuals @ self.S.T  # (N, m)
        signs = np.sign(z).astype(np.int8)
        signs[signs == 0] = 1  # break ties deterministically

        norms = np.linalg.norm(residuals, axis=1).astype(np.float32)  # (N,)

        if single:
            return signs[0], norms[0]
        return signs, norms

    # ------------------------------------------------------------------
    # Correction at query time
    # ------------------------------------------------------------------

    def correction_batch(
        self,
        q_rotated: np.ndarray,
        sign_bits: np.ndarray,
        residual_norms: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate ⟨q', e_i⟩ for all N database vectors simultaneously.

        Parameters
        ----------
        q_rotated      : (d,) float32  rotated query  (R @ query)
        sign_bits      : (N, m) int8
        residual_norms : (N,) float32

        Returns
        -------
        corrections : (N,) float32
        """
        q_rot = np.asarray(q_rotated, dtype=np.float32)
        sq = self.S @ q_rot  # (m,)  projections of query onto JL rows

        # (N, m) @ (m,) → (N,)
        raw = sign_bits.astype(np.float32) @ sq

        # Bias correction: multiply by sqrt(π/2) * ‖e‖ / m
        return (np.sqrt(np.pi / 2) / self.m) * residual_norms * raw

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def bytes_per_vector(self) -> float:
        """Bits stored per vector: m sign bits + 1 float32 norm."""
        return (self.m + 7) // 8 + 4
