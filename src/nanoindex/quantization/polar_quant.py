"""
PolarQuant: lossless-structure vector quantization via recursive polar decomposition.

Algorithm (per vector v ∈ ℝ^d):
  1. Rotate: v' = R @ v  (R orthogonal, shared across all vectors)
  2. Pair adjacent coordinates, convert each pair to polar (r, θ)
  3. Quantize θ uniformly on [-π, π] with b bits
  4. Recursively apply the same pairing to the radii r
  5. Repeat until one scalar radius remains
  6. Store: (d-1) quantized angles + 1 float32 radius
"""

from __future__ import annotations

import numpy as np

try:
    from ._numba_kernels import polar_inner_product as _numba_ip
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


class PolarQuant:
    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.bits = bits
        self._q_max = (1 << bits) - 1  # max quantized value

        # Precompute level structure
        self.level_sizes: list[int] = []
        n = dim
        while n > 1:
            self.level_sizes.append(n)
            n = (n + 1) // 2
        self.level_sizes.append(1)

        self.level_n_pairs   = [s // 2 for s in self.level_sizes[:-1]]
        self.level_n_unpaired = [s % 2  for s in self.level_sizes[:-1]]
        self.total_angles = sum(self.level_n_pairs)  # always dim - 1

        # Cumulative angle offsets per level (for O(1) slicing)
        self.angle_offsets: list[int] = [0]
        for np_ in self.level_n_pairs:
            self.angle_offsets.append(self.angle_offsets[-1] + np_)

        # Precompute cos/sin lookup tables for all 2^bits quantized angle values.
        # This replaces per-query np.cos/np.sin calls with integer array indexing.
        q_vals = np.arange(self._q_max + 1, dtype=np.float32)
        angles_lut = q_vals / self._q_max * 2 * np.pi - np.pi
        self.cos_lut = np.cos(angles_lut).astype(np.float32)  # (2^bits,)
        self.sin_lut = np.sin(angles_lut).astype(np.float32)  # (2^bits,)

        # Pre-cast level structure arrays for Numba (int64 required by njit)
        self._nb_level_n_pairs  = np.array(self.level_n_pairs,   dtype=np.int64)
        self._nb_level_n_unp    = np.array(self.level_n_unpaired, dtype=np.int64)
        self._nb_angle_offsets  = np.array(self.angle_offsets[:-1], dtype=np.int64)
        self._nb_max_level_size = int(max(self.level_sizes[1:], default=1))

        # Random orthogonal rotation matrix (QR decomposition of a Gaussian matrix)
        rng = np.random.default_rng(seed)
        H = rng.standard_normal((dim, dim))
        self.R, _ = np.linalg.qr(H)
        self.R = self.R.astype(np.float32)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compress a batch of vectors.

        Parameters
        ----------
        vectors : (N, d) float32

        Returns
        -------
        angles_int : (N, d-1) int32   quantized angles [0, 2^bits - 1]
        radii      : (N,)    float32  final scalar radii
        """
        v = np.asarray(vectors, dtype=np.float32)
        single = v.ndim == 1
        if single:
            v = v[np.newaxis]

        N = len(v)
        # Rotate
        current = v @ self.R.T  # (N, d)

        angles_int = np.empty((N, self.total_angles), dtype=np.int32)

        for i, (n_pairs, n_unp) in enumerate(zip(self.level_n_pairs, self.level_n_unpaired)):
            x = current[:, :2 * n_pairs:2]  # (N, n_pairs)  even indices
            y = current[:, 1:2 * n_pairs:2] # (N, n_pairs)  odd  indices
            unpaired = current[:, 2 * n_pairs:]  # (N, 0 or 1)

            r     = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)

            # Uniform quantization: [-π, π] → [0, q_max]
            q = np.round((theta + np.pi) / (2 * np.pi) * self._q_max).astype(np.int32)
            q = np.clip(q, 0, self._q_max)

            col = self.angle_offsets[i]
            angles_int[:, col:col + n_pairs] = q

            current = np.hstack([r, unpaired]) if n_unp else r

        radii = current[:, 0].astype(np.float32)

        if single:
            return angles_int[0], radii[0]
        return angles_int, radii

    # ------------------------------------------------------------------
    # Reconstruction (used to compute QJL residual)
    # ------------------------------------------------------------------

    def reconstruct(self, angles_int: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate vectors in the *original* (non-rotated) space.

        Parameters
        ----------
        angles_int : (N, d-1) int32
        radii      : (N,) float32

        Returns
        -------
        v_approx : (N, d) float32
        """
        N = len(radii)
        current = radii[:, np.newaxis].astype(np.float32)  # (N, 1)

        for level_idx in reversed(range(len(self.level_n_pairs))):
            n_pairs = self.level_n_pairs[level_idx]
            n_unp   = self.level_n_unpaired[level_idx]
            offset  = self.angle_offsets[level_idx]

            level_idx_arr   = angles_int[:, offset:offset + n_pairs]  # (N, n_pairs) int32
            compressed_r    = current[:, :n_pairs]                     # (N, n_pairs)
            passed_through  = current[:, n_pairs:]                     # (N, 0 or 1)

            cos_a = self.cos_lut[level_idx_arr]
            sin_a = self.sin_lut[level_idx_arr]

            # Expand each compressed radius back into its original pair
            expanded = np.empty((N, 2 * n_pairs), dtype=np.float32)
            expanded[:, 0::2] = compressed_r * cos_a  # x-components
            expanded[:, 1::2] = compressed_r * sin_a  # y-components

            current = np.hstack([expanded, passed_through]) if n_unp else expanded

        # current is (N, d) in the rotated space; rotate back
        return current @ self.R  # (N, d) in original space

    # ------------------------------------------------------------------
    # Inner product (the core operation — no full decompression)
    # ------------------------------------------------------------------

    def inner_product_batch(
        self,
        query: np.ndarray,
        angles_int: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        """
        Compute approximate inner products between one query and N compressed vectors.

        The key insight: the inner product in polar coordinates can be computed
        recursively without reconstructing the original vector.

        For each pair (v'_2i, v'_2i+1) with quantized angle θ̂_i and radius r_i:
          contribution = r_i * (q'_2i * cos(θ̂_i) + q'_2i+1 * sin(θ̂_i))

        The r_i are themselves compressed at the next level, so we recurse.

        Parameters
        ----------
        query      : (d,) float32
        angles_int : (N, d-1) int32
        radii      : (N,) float32

        Returns
        -------
        scores : (N,) float32
        """
        q_rot = (self.R @ query).astype(np.float32)  # rotate query once

        if _NUMBA_AVAILABLE:
            return _numba_ip(
                q_rot, angles_int, radii,
                self.cos_lut, self.sin_lut,
                self._nb_level_n_pairs, self._nb_level_n_unp,
                self._nb_angle_offsets, self._nb_max_level_size,
            )

        # NumPy fallback
        N = len(radii)
        current_q = None

        for level_idx, (n_pairs, n_unp) in enumerate(zip(self.level_n_pairs, self.level_n_unpaired)):
            offset    = self.angle_offsets[level_idx]
            level_ints = angles_int[:, offset:offset + n_pairs]
            cos_a = self.cos_lut[level_ints]
            sin_a = self.sin_lut[level_ints]

            if level_idx == 0:
                qx = q_rot[:2 * n_pairs:2]
                qy = q_rot[1:2 * n_pairs:2]
                A  = qx * cos_a + qy * sin_a
                if n_unp:
                    current_q = np.hstack([A, np.full((N, 1), q_rot[-1], dtype=np.float32)])
                else:
                    current_q = A
            else:
                qx    = current_q[:, :2 * n_pairs:2]
                qy    = current_q[:, 1:2 * n_pairs:2]
                unp_q = current_q[:, 2 * n_pairs:]
                A = qx * cos_a + qy * sin_a
                current_q = np.hstack([A, unp_q]) if n_unp else A

        return (radii * current_q[:, 0]).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def bytes_per_vector(self) -> float:
        """Theoretical bytes per vector at self.bits bits per angle."""
        angle_bits = self.total_angles * self.bits
        return (angle_bits + 7) // 8 + 4  # packed angles + float32 radius
