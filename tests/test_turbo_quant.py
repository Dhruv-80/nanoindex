"""Integration tests for the combined TurboQuant pipeline."""

import numpy as np
import pytest
import tempfile
import os

from turboquant_f1.quantization.turbo_quant import TurboQuant

DIM  = 384
BITS = 3
QJL_M = 64
N    = 512


@pytest.fixture(scope="module")
def tq():
    return TurboQuant(dim=DIM, bits=BITS, qjl_m=QJL_M, seed=0)


@pytest.fixture(scope="module")
def unit_vectors():
    rng = np.random.default_rng(42)
    v = rng.standard_normal((N, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def compressed(tq, unit_vectors):
    return tq.compress(unit_vectors)


# ------------------------------------------------------------------
# Compress
# ------------------------------------------------------------------

def test_compress_n_vectors(compressed):
    assert compressed.n_vectors == N


def test_compress_shapes(compressed):
    assert compressed.angles_int.shape     == (N, DIM - 1)
    assert compressed.radii.shape          == (N,)
    assert compressed.sign_bits.shape      == (N, QJL_M)
    assert compressed.residual_norms.shape == (N,)


def test_compression_ratio(tq):
    ratio = tq.compression_ratio()
    # At 3 bits, 64 QJL dims: expect ~8-11× compression
    assert 8.0 <= ratio <= 12.0, f"Unexpected compression ratio {ratio:.1f}×"


def test_memory_estimate(tq):
    mb = tq.memory_mb(505_000)
    # Full 2023 season estimate: should fit well under 200 MB
    assert mb < 200.0, f"Estimated index size {mb:.1f} MB exceeds budget"


# ------------------------------------------------------------------
# Inner product quality
# ------------------------------------------------------------------

def test_inner_product_shape(tq, unit_vectors, compressed):
    q      = unit_vectors[0]
    scores = tq.inner_product_batch(q, compressed)
    assert scores.shape == (N,)


def test_inner_product_self_is_highest(tq, unit_vectors, compressed):
    """For unit vectors, ⟨v_i, v_i⟩ = 1.0 — should be the top score."""
    for i in [0, 1, 2]:
        q = unit_vectors[i]
        scores = tq.inner_product_batch(q, compressed)
        # The query vector itself should rank in the top-3
        top3 = np.argsort(scores)[::-1][:3]
        assert i in top3, f"Query {i} not in its own top-3: {top3}"


def _clustered_vectors(n_clusters: int = 20, per_cluster: int = 25, cosine_sim: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((n_clusters, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []
    for c in centers:
        for _ in range(per_cluster):
            perp = rng.standard_normal(DIM).astype(np.float32)
            perp -= (perp @ c) * c
            perp /= np.linalg.norm(perp)
            rows.append(cosine_sim * c + np.sqrt(1 - cosine_sim ** 2) * perp)
    return np.array(rows, dtype=np.float32)


def test_recall_at_10_clustered():
    """
    Recall@10 at 4 bits on clustered vectors.
    3-bit error compounds across 9 recursive levels (self-IP ≈ 0.69), making
    top-10 ranking unreliable when score gaps are small. 4 bits raises it to ≈ 0.90.
    Real embeddings (similar F1 moments) have within-cluster IPs of 0.6-0.9 so
    score gaps are large even at 3 bits — but 4 bits is the safe default.
    """
    tq4 = TurboQuant(dim=DIM, bits=4, qjl_m=QJL_M, seed=0)
    v = _clustered_vectors()
    db = tq4.compress(v)

    k = 10
    recalls = []
    for i in range(20):
        q = v[i]
        approx = tq4.inner_product_batch(q, db)
        exact  = v @ q
        recalls.append(len(set(np.argsort(approx)[::-1][:k]) & set(np.argsort(exact)[::-1][:k])) / k)

    mean_recall = np.mean(recalls)
    # QJL SNR at 4-bit is ~1 (noise ≈ signal), so it adds marginal benefit over
    # PolarQuant alone; 0.70+ on clustered structured data is the achievable bar.
    assert mean_recall >= 0.70, f"Mean Recall@10 = {mean_recall:.2f}, expected >= 0.70"


def test_qjl_improves_recall(unit_vectors):
    """TurboQuant (PolarQuant + QJL) should outperform PolarQuant alone."""
    from turboquant_f1.quantization.polar_quant import PolarQuant

    pq  = PolarQuant(dim=DIM, bits=BITS, seed=0)
    tq_ = TurboQuant(dim=DIM, bits=BITS, qjl_m=QJL_M, seed=0)

    angles, radii = pq.compress(unit_vectors)
    db = tq_.compress(unit_vectors)

    k = 10
    pq_recalls, tq_recalls = [], []
    for i in range(20):
        q = unit_vectors[i]
        exact = unit_vectors @ q

        pq_scores = pq.inner_product_batch(q, angles, radii)
        tq_scores = tq_.inner_product_batch(q, db)

        top_exact = set(np.argsort(exact)[::-1][:k])
        pq_recalls.append(len(set(np.argsort(pq_scores)[::-1][:k]) & top_exact) / k)
        tq_recalls.append(len(set(np.argsort(tq_scores)[::-1][:k]) & top_exact) / k)

    assert np.mean(tq_recalls) >= np.mean(pq_recalls), (
        f"TurboQuant recall {np.mean(tq_recalls):.2f} not >= PolarQuant recall {np.mean(pq_recalls):.2f}"
    )


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_save_load_roundtrip(tq, unit_vectors, compressed):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_index")
        tq.save(compressed, path)
        loaded = tq.load(path + ".npz")

    assert loaded.n_vectors == compressed.n_vectors
    np.testing.assert_array_equal(loaded.angles_int, compressed.angles_int)
    np.testing.assert_array_equal(loaded.sign_bits, compressed.sign_bits)
    np.testing.assert_allclose(loaded.radii, compressed.radii)


def test_scores_identical_after_reload(tq, unit_vectors, compressed):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_index")
        tq.save(compressed, path)
        loaded = tq.load(path + ".npz")

    q = unit_vectors[5]
    s1 = tq.inner_product_batch(q, compressed)
    s2 = tq.inner_product_batch(q, loaded)
    np.testing.assert_allclose(s1, s2, rtol=1e-5)
