"""Unit tests for QJL residual correction."""

import numpy as np
import pytest

from turboquant_f1.quantization.qjl import QJL

DIM = 384
M   = 64
N   = 256


@pytest.fixture(scope="module")
def qjl():
    return QJL(dim=DIM, m=M, seed=0)


@pytest.fixture(scope="module")
def residuals():
    rng = np.random.default_rng(7)
    return rng.standard_normal((N, DIM)).astype(np.float32) * 0.1


# ------------------------------------------------------------------
# compress_residuals
# ------------------------------------------------------------------

def test_compress_shapes(qjl, residuals):
    signs, norms = qjl.compress_residuals(residuals)
    assert signs.shape == (N, M)
    assert norms.shape == (N,)


def test_sign_bits_values(qjl, residuals):
    signs, _ = qjl.compress_residuals(residuals)
    assert set(np.unique(signs)).issubset({-1, 1})


def test_residual_norms_nonneg(qjl, residuals):
    _, norms = qjl.compress_residuals(residuals)
    assert (norms >= 0).all()


def test_compress_single_residual(qjl, residuals):
    signs, norm = qjl.compress_residuals(residuals[0])
    assert signs.shape == (M,)


# ------------------------------------------------------------------
# correction_batch — estimator quality
# ------------------------------------------------------------------

def test_correction_shape(qjl, residuals):
    rng = np.random.default_rng(99)
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)

    signs, norms = qjl.compress_residuals(residuals)
    corr = qjl.correction_batch(q, signs, norms)
    assert corr.shape == (N,)


def test_correction_reduces_error(qjl):
    """Adding the QJL correction should reduce mean absolute error vs. exact ⟨q, e⟩."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal((512, DIM)).astype(np.float32) * 0.2
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)

    exact_dots = residuals @ q  # (512,)

    signs, norms = qjl.compress_residuals(residuals)
    corrections  = qjl.correction_batch(q, signs, norms)

    # Without correction: estimate is 0 for residuals (PolarQuant alone)
    mae_no_corr = np.abs(exact_dots).mean()
    mae_with_corr = np.abs(corrections - exact_dots).mean()

    assert mae_with_corr < mae_no_corr, (
        f"Correction did not reduce MAE: {mae_with_corr:.4f} vs {mae_no_corr:.4f}"
    )


def test_correction_unbiased(qjl):
    """E[correction] ≈ E[⟨q, e⟩] for large m."""
    rng = np.random.default_rng(0)
    n = 2000
    residuals = rng.standard_normal((n, DIM)).astype(np.float32) * 0.15
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)

    exact_mean = (residuals @ q).mean()

    signs, norms = qjl.compress_residuals(residuals)
    corr_mean   = qjl.correction_batch(q, signs, norms).mean()

    assert abs(corr_mean - exact_mean) < 0.02, (
        f"Bias too large: estimated {corr_mean:.4f}, exact {exact_mean:.4f}"
    )
