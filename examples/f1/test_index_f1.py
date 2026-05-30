"""Unit tests for TurboQuantIndex — no I/O, no FastF1, no sentence-transformers."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from turboquant_f1.index.filters import apply_filters
from turboquant_f1.index.turbo_index import TurboQuantIndex, SearchResult

DIM = 384
N   = 200


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_unit_vectors(n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _make_chunks(n: int = N) -> list[dict]:
    drivers = ["VER", "HAM", "LEC", "NOR", "SAI"]
    chunk_types = ["telemetry", "lap_summary", "event"]
    return [
        {
            "id": f"chunk_{i:04d}",
            "text": f"Sample chunk {i} text describing F1 moment.",
            "chunk_type": chunk_types[i % 3],
            "year": 2023,
            "race": "bahrain_grand_prix" if i < N // 2 else "jeddah_grand_prix",
            "session": "R",
            "driver": drivers[i % 5],
            "lap": (i % 57) + 1,
            "timestamp_start": float(i * 5),
            "timestamp_end": float(i * 5 + 5),
            "compound": "SOFT" if i % 3 == 0 else "MEDIUM",
            "position": (i % 20) + 1,
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def idx():
    index = TurboQuantIndex(dim=DIM, bits=3, qjl_m=64, seed=0)
    index.add(_make_unit_vectors(), _make_chunks())
    return index


# ------------------------------------------------------------------
# add()
# ------------------------------------------------------------------

def test_add_n_vectors(idx):
    assert idx.n_vectors == N


def test_add_meta_length(idx):
    assert len(idx._meta) == N


def test_add_mismatch_raises():
    index = TurboQuantIndex(dim=DIM, bits=3)
    with pytest.raises(ValueError):
        index.add(_make_unit_vectors(10), _make_chunks(20))


# ------------------------------------------------------------------
# search()
# ------------------------------------------------------------------

def test_search_returns_k_results(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=10)
    assert len(results) == 10


def test_search_results_are_search_result_type(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=5)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_sorted_descending(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_rank_field(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=5)
    assert [r.rank for r in results] == list(range(5))


def test_search_empty_on_empty_index():
    empty = TurboQuantIndex(dim=DIM)
    with pytest.raises(RuntimeError):
        empty.search(np.zeros(DIM, dtype=np.float32))


def test_search_k_capped_by_available(idx):
    results = idx.search(_make_unit_vectors(1)[0], k=N + 100)
    assert len(results) == N


# ------------------------------------------------------------------
# filters
# ------------------------------------------------------------------

def test_filter_by_driver(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=20, filters={"driver": "VER"})
    assert all(r.driver == "VER" for r in results)


def test_filter_by_chunk_type(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=20, filters={"chunk_type": "event"})
    assert all(r.chunk_type == "event" for r in results)


def test_filter_by_race(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=20, filters={"race": "bahrain_grand_prix"})
    assert all(r.race == "bahrain_grand_prix" for r in results)


def test_filter_by_lap_range(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=20, filters={"lap_min": 10, "lap_max": 20})
    assert all(10 <= r.lap <= 20 for r in results)


def test_filter_combined(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=10, filters={"driver": "HAM", "chunk_type": "telemetry"})
    assert all(r.driver == "HAM" and r.chunk_type == "telemetry" for r in results)


def test_filter_no_match_returns_empty(idx):
    q = _make_unit_vectors(1)[0]
    results = idx.search(q, k=10, filters={"driver": "NONEXISTENT"})
    assert results == []


# ------------------------------------------------------------------
# apply_filters() directly
# ------------------------------------------------------------------

def test_apply_filters_none_returns_none():
    meta = _make_chunks(10)
    assert apply_filters(meta, None) is None


def test_apply_filters_case_insensitive():
    meta = _make_chunks(10)
    mask = apply_filters(meta, {"driver": "ver"})
    assert mask is not None
    assert mask.sum() > 0


def test_apply_filters_list_criterion():
    meta = _make_chunks(20)
    mask = apply_filters(meta, {"driver": ["VER", "HAM"]})
    assert mask is not None
    for i, m in enumerate(meta):
        expected = m["driver"] in ("VER", "HAM")
        assert mask[i] == expected


# ------------------------------------------------------------------
# save / load
# ------------------------------------------------------------------

def test_save_creates_files(idx):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_idx")
        idx.save(path)
        assert os.path.exists(path + ".npz")
        assert os.path.exists(path + ".meta.json")


def test_load_roundtrip(idx):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_idx")
        idx.save(path)
        loaded = TurboQuantIndex.load(path)

    assert loaded.n_vectors == idx.n_vectors
    assert loaded.dim == idx.dim
    assert loaded.bits == idx.bits


def test_search_identical_after_reload(idx):
    q = _make_unit_vectors(1, seed=99)[0]
    scores_orig = [r.score for r in idx.search(q, k=10)]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_idx")
        idx.save(path)
        loaded = TurboQuantIndex.load(path)

    scores_loaded = [r.score for r in loaded.search(q, k=10)]
    np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)


def test_load_with_npz_extension(idx):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_idx")
        idx.save(path)
        loaded = TurboQuantIndex.load(path + ".npz")
    assert loaded.n_vectors == idx.n_vectors


# ------------------------------------------------------------------
# stats()
# ------------------------------------------------------------------

def test_stats_keys(idx):
    s = idx.stats()
    for key in ("n_vectors", "dim", "bits", "chunk_types", "memory_mb", "compression_ratio"):
        assert key in s


def test_stats_memory_reasonable(idx):
    s = idx.stats()
    # N=200 vectors at 3-bit: < 1 MB
    assert s["memory_mb"] < 1.0


def test_stats_chunk_types_complete(idx):
    s = idx.stats()
    assert set(s["chunk_types"].keys()) == {"telemetry", "lap_summary", "event"}
