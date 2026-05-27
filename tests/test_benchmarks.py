"""Smoke tests for the benchmark suite — no Faiss I/O, no embeddings file required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.baselines import brute_force, faiss_pq, faiss_sq

DIM = 64
N   = 256


@pytest.fixture(scope="module")
def unit_vecs():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((N, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------------
# brute_force
# ------------------------------------------------------------------

def test_brute_force_returns_k(unit_vecs):
    idx, scores = brute_force.search(unit_vecs[0], unit_vecs, k=10)
    assert len(idx) == 10
    assert len(scores) == 10


def test_brute_force_sorted(unit_vecs):
    _, scores = brute_force.search(unit_vecs[0], unit_vecs, k=10)
    assert list(scores) == sorted(scores, reverse=True)


def test_brute_force_self_is_top(unit_vecs):
    idx, _ = brute_force.search(unit_vecs[0], unit_vecs, k=1)
    assert idx[0] == 0


# ------------------------------------------------------------------
# faiss_pq
# ------------------------------------------------------------------

def test_faiss_pq_builds(unit_vecs):
    index = faiss_pq.build(unit_vecs, m=8, bits=8)
    assert index.ntotal == N


def test_faiss_pq_search_returns_k(unit_vecs):
    index = faiss_pq.build(unit_vecs, m=8, bits=8)
    idx, scores = faiss_pq.search(index, unit_vecs[0], k=10)
    assert len(idx) == 10


def test_faiss_pq_memory_bytes(unit_vecs):
    index = faiss_pq.build(unit_vecs, m=8, bits=8)
    mb = faiss_pq.memory_bytes(index)
    assert mb == N * 8  # 8 sub-quantizers × 1 byte each


# ------------------------------------------------------------------
# faiss_sq
# ------------------------------------------------------------------

def test_faiss_sq_builds(unit_vecs):
    index = faiss_sq.build(unit_vecs)
    assert index.ntotal == N


def test_faiss_sq_search_returns_k(unit_vecs):
    index = faiss_sq.build(unit_vecs)
    idx, scores = faiss_sq.search(index, unit_vecs[0], k=10)
    assert len(idx) == 10


def test_faiss_sq_memory_bytes(unit_vecs):
    index = faiss_sq.build(unit_vecs)
    mb = faiss_sq.memory_bytes(index)
    assert mb == N * DIM  # SQ8: 1 byte per dimension


# ------------------------------------------------------------------
# recall helper
# ------------------------------------------------------------------

def test_brute_force_is_ground_truth_for_recall(unit_vecs):
    """TurboQuant recall vs brute-force — smoke test that recalls are in [0,1]."""
    from turboquant_f1.quantization.turbo_quant import TurboQuant

    tq = TurboQuant(dim=DIM, bits=4, qjl_m=16, seed=0)
    db = tq.compress(unit_vecs)

    k = 10
    recalls = []
    for i in range(20):
        q = unit_vecs[i]
        exact_idx, _ = brute_force.search(q, unit_vecs, k)
        approx_scores = tq.inner_product_batch(q, db)
        approx_idx = np.argsort(approx_scores)[::-1][:k]
        recalls.append(len(set(approx_idx) & set(exact_idx)) / k)

    mean_recall = np.mean(recalls)
    assert 0.0 <= mean_recall <= 1.0
    assert mean_recall >= 0.3, f"Expected recall >= 0.3, got {mean_recall:.2f}"
