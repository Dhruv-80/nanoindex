"""Float32 brute-force baseline — ground truth for recall and distortion."""

from __future__ import annotations

import numpy as np


def search(query: np.ndarray, database: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-k inner product search.

    Returns (indices, scores) sorted by descending score.
    """
    scores = database @ query  # (N,)
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]
