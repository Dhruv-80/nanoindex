"""Faiss Product Quantization baseline."""

from __future__ import annotations

import numpy as np


def build(vectors: np.ndarray, m: int = 48, bits: int = 8) -> "faiss.IndexPQ":
    """
    Build a Faiss PQ index over float32 vectors.

    m    : number of sub-quantizers (must divide dim)
    bits : bits per sub-quantizer code (typically 8)
    """
    import faiss

    n, d = vectors.shape
    # m must divide d; find largest m <= requested that divides d
    while d % m != 0 and m > 1:
        m -= 1

    index = faiss.IndexPQ(d, m, bits, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def search(index, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query[None, :], k)
    return indices[0], scores[0]


def memory_bytes(index) -> int:
    return index.sa_code_size() * index.ntotal
