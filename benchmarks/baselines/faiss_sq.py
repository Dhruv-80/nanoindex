"""Faiss Scalar Quantization (SQ8) baseline."""

from __future__ import annotations

import numpy as np


def build(vectors: np.ndarray) -> "faiss.IndexScalarQuantizer":
    """Build a Faiss SQ8 index (8-bit scalar quantization)."""
    import faiss

    d = vectors.shape[1]
    index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def search(index, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query[None, :], k)
    return indices[0], scores[0]


def memory_bytes(index) -> int:
    return index.sa_code_size() * index.ntotal
