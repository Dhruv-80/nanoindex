"""Tests for NanoIndex — domain-agnostic compressed vector index."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from nanoindex import NanoIndex, SearchResult, apply_filters

DIM = 384
N   = 200


def _unit_vecs(n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _meta(n: int = N) -> list[dict]:
    return [
        {
            "id":       f"doc_{i:04d}",
            "text":     f"Document {i} about topic {i % 5}.",
            "category": ["A", "B", "C"][i % 3],
            "score":    float(i) / n,
            "year":     2020 + (i % 5),
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def idx():
    index = NanoIndex(dim=DIM, bits=4, qjl_m=64, seed=0)
    index.add(_unit_vecs(), _meta())
    return index


# ------------------------------------------------------------------
# add
# ------------------------------------------------------------------

def test_add_n_vectors(idx):
    assert idx.n_vectors == N


def test_add_mismatch_raises():
    index = NanoIndex(dim=DIM, bits=4)
    with pytest.raises(ValueError):
        index.add(_unit_vecs(10), _meta(20))


# ------------------------------------------------------------------
# search — single query
# ------------------------------------------------------------------

def test_search_returns_k(idx):
    results = idx.search(_unit_vecs(1)[0], k=10)
    assert len(results) == 10


def test_search_result_type(idx):
    results = idx.search(_unit_vecs(1)[0], k=5)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_sorted_descending(idx):
    results = idx.search(_unit_vecs(1)[0], k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_rank_field(idx):
    results = idx.search(_unit_vecs(1)[0], k=5)
    assert [r.rank for r in results] == list(range(5))


def test_search_result_has_id_text_metadata(idx):
    results = idx.search(_unit_vecs(1)[0], k=1)
    r = results[0]
    assert isinstance(r.id, str)
    assert isinstance(r.text, str)
    assert isinstance(r.metadata, dict)
    assert "id" not in r.metadata
    assert "text" not in r.metadata


def test_search_empty_index_raises():
    with pytest.raises(RuntimeError):
        NanoIndex(dim=DIM).search(np.zeros(DIM, dtype=np.float32))


def test_search_k_capped_by_n(idx):
    results = idx.search(_unit_vecs(1)[0], k=N + 100)
    assert len(results) == N


# ------------------------------------------------------------------
# search — batch query
# ------------------------------------------------------------------

def test_batch_search_returns_list_of_lists(idx):
    queries = _unit_vecs(5)
    results = idx.search(queries, k=10)
    assert isinstance(results, list)
    assert len(results) == 5
    assert all(isinstance(r, list) for r in results)
    assert all(len(r) == 10 for r in results)


def test_batch_search_each_sorted(idx):
    queries = _unit_vecs(3)
    batch = idx.search(queries, k=10)
    for results in batch:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# filters
# ------------------------------------------------------------------

def test_filter_equality(idx):
    results = idx.search(_unit_vecs(1)[0], k=20, filters={"category": "A"})
    assert all(r.metadata["category"] == "A" for r in results)


def test_filter_list_criterion(idx):
    results = idx.search(_unit_vecs(1)[0], k=30, filters={"category": ["A", "B"]})
    assert all(r.metadata["category"] in ("A", "B") for r in results)


def test_filter_range_min(idx):
    results = idx.search(_unit_vecs(1)[0], k=20, filters={"score_min": 0.5})
    assert all(r.metadata["score"] >= 0.5 for r in results)


def test_filter_range_max(idx):
    results = idx.search(_unit_vecs(1)[0], k=20, filters={"score_max": 0.3})
    assert all(r.metadata["score"] <= 0.3 for r in results)


def test_filter_combined(idx):
    results = idx.search(_unit_vecs(1)[0], k=10,
                         filters={"category": "A", "year_min": 2022})
    assert all(r.metadata["category"] == "A" and r.metadata["year"] >= 2022
               for r in results)


def test_filter_no_match_returns_empty(idx):
    results = idx.search(_unit_vecs(1)[0], k=10, filters={"category": "NONEXISTENT"})
    assert results == []


# ------------------------------------------------------------------
# apply_filters directly
# ------------------------------------------------------------------

def test_apply_filters_none_returns_none():
    assert apply_filters(_meta(10), None) is None


def test_apply_filters_case_insensitive():
    mask = apply_filters(_meta(10), {"category": "a"})
    assert mask is not None and mask.sum() > 0


def test_apply_filters_range():
    meta = _meta(100)
    mask = apply_filters(meta, {"score_min": 0.4, "score_max": 0.6})
    for i, m in enumerate(meta):
        assert mask[i] == (0.4 <= m["score"] <= 0.6)


# ------------------------------------------------------------------
# save / load
# ------------------------------------------------------------------

def test_save_creates_files(idx):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "idx")
        idx.save(path)
        assert os.path.exists(path + ".npz")
        assert os.path.exists(path + ".meta.json")


def test_load_roundtrip(idx):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "idx")
        idx.save(path)
        loaded = NanoIndex.load(path)
    assert loaded.n_vectors == idx.n_vectors
    assert loaded.dim == idx.dim
    assert loaded.bits == idx.bits


def test_search_identical_after_reload(idx):
    q = _unit_vecs(1, seed=99)[0]
    orig = [r.score for r in idx.search(q, k=10)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "idx")
        idx.save(path)
        loaded = NanoIndex.load(path)
    reloaded = [r.score for r in loaded.search(q, k=10)]
    np.testing.assert_allclose(orig, reloaded, rtol=1e-5)


def test_load_with_npz_extension(idx):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "idx")
        idx.save(path)
        loaded = NanoIndex.load(path + ".npz")
    assert loaded.n_vectors == N


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

def test_stats_keys(idx):
    s = idx.stats()
    for key in ("n_vectors", "dim", "bits", "memory_mb", "compression_ratio"):
        assert key in s


def test_stats_memory_reasonable(idx):
    assert idx.stats()["memory_mb"] < 1.0


def test_stats_compression_ratio(idx):
    assert idx.stats()["compression_ratio"] >= 5.0
