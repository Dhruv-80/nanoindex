"""
TurboQuant F1 — automated benchmark suite.

Measures and compares:
  - TurboQuant (this project)
  - Faiss PQ (product quantization)
  - Faiss SQ8 (scalar quantization)
  - Brute-force float32 (ground truth)

Outputs: benchmarks/reports/results.json + printed markdown table.

Usage:
    python benchmarks/run_benchmarks.py [--embeddings path] [--n-queries 200] [--bits 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from project root or benchmarks/
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from turboquant_f1.quantization.turbo_quant import TurboQuant
from benchmarks.baselines import brute_force, faiss_pq, faiss_sq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_embeddings(path: str) -> np.ndarray:
    data = np.load(path)
    emb = data["embeddings"].astype(np.float32)
    # L2-normalise (should already be, but ensure it)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)


def _recall_at_k(approx_idx: np.ndarray, exact_idx: np.ndarray, k: int) -> float:
    return len(set(approx_idx[:k]) & set(exact_idx[:k])) / k


def _latency_stats(times_s: list[float]) -> dict:
    arr = np.array(times_s) * 1000  # ms
    return {
        "p50_ms":  round(float(np.percentile(arr, 50)), 2),
        "p95_ms":  round(float(np.percentile(arr, 95)), 2),
        "p99_ms":  round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
    }


def _ip_distortion(exact_scores: np.ndarray, approx_scores: np.ndarray) -> float:
    """Mean squared error between exact and approximate inner products."""
    return float(np.mean((exact_scores - approx_scores) ** 2))


# ---------------------------------------------------------------------------
# Per-method benchmark
# ---------------------------------------------------------------------------

def _bench_turbo(db: np.ndarray, queries: np.ndarray, bits: int, qjl_m: int, ks: list[int]) -> dict:
    dim = db.shape[1]
    tq = TurboQuant(dim=dim, bits=bits, qjl_m=qjl_m, seed=42)

    t0 = time.perf_counter()
    compressed = tq.compress(db)
    build_s = time.perf_counter() - t0

    # Memory
    bytes_per_vec = tq.polar.bytes_per_vector() + tq.qjl.bytes_per_vector()
    compressed_bytes = bytes_per_vec * len(db)
    float32_bytes = db.nbytes
    compression_ratio = float32_bytes / compressed_bytes

    # Recall and distortion
    recalls = {k: [] for k in ks}
    distortions = []
    latencies = []

    for q in queries:
        exact_idx, exact_scores = brute_force.search(q, db, max(ks))

        t0 = time.perf_counter()
        approx_scores = tq.inner_product_batch(q, compressed)
        latencies.append(time.perf_counter() - t0)

        approx_idx = np.argsort(approx_scores)[::-1]
        for k in ks:
            recalls[k].append(_recall_at_k(approx_idx, exact_idx, k))

        distortions.append(_ip_distortion(exact_scores, approx_scores[exact_idx]))

    return {
        "method": f"TurboQuant (bits={bits}, qjl_m={qjl_m})",
        "build_s": round(build_s, 2),
        "compressed_mb": round(compressed_bytes / 1e6, 2),
        "float32_mb": round(float32_bytes / 1e6, 2),
        "compression_ratio": round(compression_ratio, 2),
        "recall": {f"@{k}": round(float(np.mean(recalls[k])), 4) for k in ks},
        "ip_mse": round(float(np.mean(distortions)), 6),
        "latency": _latency_stats(latencies),
    }


def _bench_faiss_pq(db: np.ndarray, queries: np.ndarray, ks: list[int]) -> dict:
    import faiss as _faiss

    t0 = time.perf_counter()
    index = faiss_pq.build(db)
    build_s = time.perf_counter() - t0

    compressed_bytes = faiss_pq.memory_bytes(index)
    compression_ratio = db.nbytes / compressed_bytes

    recalls = {k: [] for k in ks}
    distortions = []
    latencies = []

    for q in queries:
        exact_idx, exact_scores = brute_force.search(q, db, max(ks))

        t0 = time.perf_counter()
        approx_idx, approx_scores = faiss_pq.search(index, q, max(ks))
        latencies.append(time.perf_counter() - t0)

        for k in ks:
            recalls[k].append(_recall_at_k(approx_idx, exact_idx, k))

        # distortion: compare top-k approximate scores vs exact inner products
        approx_ip = db[approx_idx[:len(exact_scores)]] @ q
        distortions.append(_ip_distortion(exact_scores, approx_ip))

    return {
        "method": "Faiss PQ (m=48, 8bit)",
        "build_s": round(build_s, 2),
        "compressed_mb": round(compressed_bytes / 1e6, 2),
        "float32_mb": round(db.nbytes / 1e6, 2),
        "compression_ratio": round(compression_ratio, 2),
        "recall": {f"@{k}": round(float(np.mean(recalls[k])), 4) for k in ks},
        "ip_mse": round(float(np.mean(distortions)), 6),
        "latency": _latency_stats(latencies),
    }


def _bench_faiss_sq(db: np.ndarray, queries: np.ndarray, ks: list[int]) -> dict:
    t0 = time.perf_counter()
    index = faiss_sq.build(db)
    build_s = time.perf_counter() - t0

    compressed_bytes = faiss_sq.memory_bytes(index)
    compression_ratio = db.nbytes / compressed_bytes

    recalls = {k: [] for k in ks}
    distortions = []
    latencies = []

    for q in queries:
        exact_idx, exact_scores = brute_force.search(q, db, max(ks))

        t0 = time.perf_counter()
        approx_idx, approx_scores = faiss_sq.search(index, q, max(ks))
        latencies.append(time.perf_counter() - t0)

        for k in ks:
            recalls[k].append(_recall_at_k(approx_idx, exact_idx, k))

        approx_ip = db[approx_idx[:len(exact_scores)]] @ q
        distortions.append(_ip_distortion(exact_scores, approx_ip))

    return {
        "method": "Faiss SQ8",
        "build_s": round(build_s, 2),
        "compressed_mb": round(compressed_bytes / 1e6, 2),
        "float32_mb": round(db.nbytes / 1e6, 2),
        "compression_ratio": round(compression_ratio, 2),
        "recall": {f"@{k}": round(float(np.mean(recalls[k])), 4) for k in ks},
        "ip_mse": round(float(np.mean(distortions)), 6),
        "latency": _latency_stats(latencies),
    }


def _bench_brute_force(db: np.ndarray, queries: np.ndarray, ks: list[int]) -> dict:
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        brute_force.search(q, db, max(ks))
        latencies.append(time.perf_counter() - t0)

    return {
        "method": "Brute-force float32",
        "build_s": 0.0,
        "compressed_mb": round(db.nbytes / 1e6, 2),
        "float32_mb": round(db.nbytes / 1e6, 2),
        "compression_ratio": 1.0,
        "recall": {f"@{k}": 1.0 for k in ks},
        "ip_mse": 0.0,
        "latency": _latency_stats(latencies),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], ks: list[int]) -> None:
    cols = ["Method", "Ratio", "MB"] + [f"R@{k}" for k in ks] + ["MSE", "p50ms", "p95ms"]
    widths = [28, 6, 7] + [6] * len(ks) + [10, 7, 7]

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep    = "  ".join("-" * w for w in widths)
    print("\n" + header)
    print(sep)
    for r in results:
        row = [
            r["method"][:28],
            f"{r['compression_ratio']:.1f}×",
            f"{r['compressed_mb']:.1f}",
        ] + [
            f"{r['recall'][f'@{k}']:.3f}" for k in ks
        ] + [
            f"{r['ip_mse']:.5f}",
            f"{r['latency']['p50_ms']:.1f}",
            f"{r['latency']['p95_ms']:.1f}",
        ]
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant F1 benchmark suite")
    parser.add_argument("--embeddings", default="data/raw/2023/bahrain_grand_prix/r/embeddings.npz",
                        help="Path to embeddings.npz")
    parser.add_argument("--n-queries", type=int, default=200, help="Number of query vectors")
    parser.add_argument("--bits", type=int, default=3, help="TurboQuant bits (2–8)")
    parser.add_argument("--qjl-m", type=int, default=64, help="QJL projection dimensions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/reports/results.json")
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings} …")
    db = _load_embeddings(args.embeddings)
    n, dim = db.shape
    print(f"  {n:,} vectors, dim={dim}")

    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(n, size=min(args.n_queries, n), replace=False)
    queries = db[q_idx]
    # Remove queries from db to avoid trivial self-matches
    mask = np.ones(n, dtype=bool)
    mask[q_idx] = False
    db_eval = db[mask]
    print(f"  {len(db_eval):,} database vectors, {len(queries)} queries\n")

    ks = [1, 10, 100]
    results = []

    print("Benchmarking brute-force float32 …")
    results.append(_bench_brute_force(db_eval, queries, ks))

    print("Benchmarking TurboQuant …")
    results.append(_bench_turbo(db_eval, queries, args.bits, args.qjl_m, ks))

    print("Benchmarking Faiss PQ …")
    results.append(_bench_faiss_pq(db_eval, queries, ks))

    print("Benchmarking Faiss SQ8 …")
    results.append(_bench_faiss_sq(db_eval, queries, ks))

    _print_table(results, ks)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"n_vectors": len(db_eval), "dim": dim, "n_queries": len(queries),
                   "results": results}, f, indent=2)
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
