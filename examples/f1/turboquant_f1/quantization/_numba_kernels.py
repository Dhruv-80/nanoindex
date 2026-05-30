"""
Numba-accelerated inner product kernel for PolarQuant.

Isolated here so polar_quant.py has a clean NumPy-only fallback path
when Numba is unavailable.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True)
def polar_inner_product(
    q_rot: np.ndarray,          # (d,) float32 — pre-rotated query
    angles_int: np.ndarray,     # (N, total_angles) int32
    radii: np.ndarray,          # (N,) float32
    cos_lut: np.ndarray,        # (2^bits,) float32
    sin_lut: np.ndarray,        # (2^bits,) float32
    level_n_pairs: np.ndarray,  # (n_levels,) int64
    level_n_unp: np.ndarray,    # (n_levels,) int64
    angle_offsets: np.ndarray,  # (n_levels,) int64
    max_level_size: int,
) -> np.ndarray:
    """
    Compute approximate inner products between a single query and N compressed
    vectors entirely in the compressed domain.

    Each iteration of the outer prange loop is independent — no data races.
    Numba allocates `buf` per-thread on the stack.
    """
    N = radii.shape[0]
    n_levels = level_n_pairs.shape[0]
    scores = np.empty(N, np.float32)

    for i in prange(N):
        buf = np.empty(max_level_size, np.float32)

        for lv in range(n_levels):
            n_pairs = level_n_pairs[lv]
            n_unp   = level_n_unp[lv]
            offset  = angle_offsets[lv]

            if lv == 0:
                for j in range(n_pairs):
                    qx = q_rot[2 * j]
                    qy = q_rot[2 * j + 1]
                    cos_a = cos_lut[angles_int[i, offset + j]]
                    sin_a = sin_lut[angles_int[i, offset + j]]
                    buf[j] = qx * cos_a + qy * sin_a
                if n_unp:
                    buf[n_pairs] = q_rot[2 * n_pairs]
            else:
                for j in range(n_pairs):
                    qx = buf[2 * j]
                    qy = buf[2 * j + 1]
                    cos_a = cos_lut[angles_int[i, offset + j]]
                    sin_a = sin_lut[angles_int[i, offset + j]]
                    buf[j] = qx * cos_a + qy * sin_a
                if n_unp:
                    buf[n_pairs] = buf[2 * n_pairs]

        scores[i] = radii[i] * buf[0]

    return scores
