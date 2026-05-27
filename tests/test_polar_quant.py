"""Unit tests for PolarQuant — no I/O, no FastF1."""

import numpy as np
import pytest

from nanoindex.quantization.polar_quant import PolarQuant

DIM = 384
BITS = 3
N = 512


@pytest.fixture(scope="module")
def pq():
    return PolarQuant(dim=DIM, bits=BITS, seed=0)


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(42)
    v = rng.standard_normal((N, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------

def test_total_angles(pq):
    assert pq.total_angles == DIM - 1


def test_level_sizes_decreasing(pq):
    for a, b in zip(pq.level_sizes, pq.level_sizes[1:]):
        assert b < a


def test_angle_offsets_end(pq):
    assert pq.angle_offsets[-1] == pq.total_angles


# ------------------------------------------------------------------
# Compress output shape and range
# ------------------------------------------------------------------

def test_compress_shapes(pq, vectors):
    angles, radii = pq.compress(vectors)
    assert angles.shape == (N, DIM - 1)
    assert radii.shape == (N,)


def test_compress_angles_in_range(pq, vectors):
    angles, _ = pq.compress(vectors)
    assert angles.min() >= 0
    assert angles.max() <= (1 << BITS) - 1


def test_compress_radii_nonneg(pq, vectors):
    _, radii = pq.compress(vectors)
    assert (radii >= 0).all()


def test_compress_radii_approx_norm(pq, vectors):
    # The final radius should approximate ‖v‖ = 1 (unit vectors)
    _, radii = pq.compress(vectors)
    assert np.allclose(radii, 1.0, atol=1e-4), f"max deviation: {np.abs(radii - 1).max()}"


def test_compress_single_vector(pq, vectors):
    angles, radius = pq.compress(vectors[0])
    assert angles.shape == (DIM - 1,)
    assert np.isscalar(radius) or radius.shape == ()


# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------

def test_reconstruct_shape(pq, vectors):
    angles, radii = pq.compress(vectors)
    recon = pq.reconstruct(angles, radii)
    assert recon.shape == (N, DIM)


def test_reconstruct_error_decreases_with_bits():
    """Higher bit-width should give lower reconstruction MSE."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((256, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    mse_prev = np.inf
    for bits in [1, 2, 3, 4]:
        pq_ = PolarQuant(dim=DIM, bits=bits, seed=0)
        angles, radii = pq_.compress(v)
        recon = pq_.reconstruct(angles, radii)
        mse = np.mean((v - recon) ** 2)
        assert mse < mse_prev, f"MSE did not decrease from bits={bits-1} to bits={bits}"
        mse_prev = mse


def test_reconstruct_mse_reasonable(pq, vectors):
    angles, radii = pq.compress(vectors)
    recon = pq.reconstruct(angles, radii)
    mse = np.mean((vectors - recon) ** 2)
    # For unit vectors at 3 bits, MSE should be well under 0.05
    assert mse < 0.05, f"MSE {mse:.4f} too high for {BITS}-bit compression"


# ------------------------------------------------------------------
# Inner product approximation
# ------------------------------------------------------------------

def test_inner_product_batch_shape(pq, vectors):
    angles, radii = pq.compress(vectors)
    q = vectors[0]
    scores = pq.inner_product_batch(q, angles, radii)
    assert scores.shape == (N,)


def test_inner_product_exact_for_uncompressed():
    """At very high bits, polar IP should match exact dot product closely."""
    pq_hi = PolarQuant(dim=DIM, bits=8, seed=0)
    rng = np.random.default_rng(1)
    v = rng.standard_normal((128, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)

    angles, radii = pq_hi.compress(v)
    approx = pq_hi.inner_product_batch(q, angles, radii)
    exact  = v @ q

    mae = np.abs(approx - exact).mean()
    assert mae < 0.01, f"MAE {mae:.4f} too high at 8 bits"


def _clustered_vectors(
    n_clusters: int = 20, per_cluster: int = 25, cosine_sim: float = 0.85, seed: int = 0
) -> np.ndarray:
    """
    Vectors where each has exactly `cosine_sim` similarity to its cluster center.
    Within-cluster pair IP ≈ cosine_sim² ≈ 0.72; across-cluster ≈ 0.
    noise=0.25 fails in 384D because noise L2 norm ≈ 4.9 >> center norm 1.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    rows = []
    for c in centers:
        for _ in range(per_cluster):
            perp = rng.standard_normal(DIM).astype(np.float32)
            perp -= (perp @ c) * c       # remove center component
            perp /= np.linalg.norm(perp)
            v = cosine_sim * c + np.sqrt(1 - cosine_sim ** 2) * perp
            rows.append(v)
    return np.array(rows, dtype=np.float32)


def test_inner_product_recall_at_10():
    """Recall@10 on clustered vectors at 4 bits (3-bit error compounds across 9 levels)."""
    pq4 = PolarQuant(dim=DIM, bits=4, seed=0)
    v = _clustered_vectors()
    angles, radii = pq4.compress(v)

    k = 10
    recalls = []
    for i in range(20):
        q = v[i]
        approx = pq4.inner_product_batch(q, angles, radii)
        exact  = v @ q
        top_approx = set(np.argsort(approx)[::-1][:k])
        top_exact  = set(np.argsort(exact)[::-1][:k])
        recalls.append(len(top_approx & top_exact) / k)

    mean_recall = np.mean(recalls)
    assert mean_recall >= 0.7, f"Recall@10 = {mean_recall:.2f}, expected >= 0.70"
