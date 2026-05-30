#!/usr/bin/env python3
"""
Production-grade NanoIndex RAG pipeline for F1 telemetry.

NanoIndex: 4-bit quantized vector index (7-10× compression).
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.
Generates or loads real F1 telemetry, chunks into ~5000 documents.
Measures index size, build time, query latency, and memory usage.
Compares against baseline RAG pipeline.

Run with: python prod_pipeline_nanoindex.py
"""

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Import NanoIndex
try:
    from nanoindex import NanoIndex
    HAS_NANOINDEX = True
except ImportError:
    HAS_NANOINDEX = False
    logger.error("NanoIndex not available; install with: pip install nanoindex-rag")

# Try to import FastF1; fall back to synthetic generation if unavailable
try:
    import fastf1
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False
    logger.warning("FastF1 not available; will use synthetic F1 telemetry data")


# ─────────────────────────────────────────────────────────────────────
# Synthetic F1 telemetry data generation (shared with baseline)
# ─────────────────────────────────────────────────────────────────────

def generate_synthetic_telemetry(
    n_documents: int = 5000,
    seed: int = 42
) -> list[dict]:
    """
    Generate synthetic F1 telemetry documents.
    Each document represents a small time window of driver telemetry.

    Parameters
    ----------
    n_documents : int
        Target number of documents to generate (~5000)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list[dict]
        List of telemetry documents with "id" and "text" keys
    """
    np.random.seed(seed)

    drivers = [
        "VER", "HAM", "LEC", "SAI", "PER", "RUS", "ALO", "STR",
        "NOR", "PIA", "HUL", "MAG", "BOT", "ZHO", "ALB",
    ]
    circuits = [
        "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
        "Monaco", "Canada", "Spain", "Austria", "Silverstone",
        "Hungary", "Belgium", "Italy", "Singapore", "Japan",
    ]
    tire_compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    weather_conditions = ["Sunny", "Cloudy", "Light Rain", "Heavy Rain", "Overcast"]

    documents = []
    doc_id = 0

    for _ in range(n_documents):
        driver = np.random.choice(drivers)
        circuit = np.random.choice(circuits)
        lap = np.random.randint(1, 60)
        sector = np.random.randint(1, 4)

        speed = np.random.randint(150, 330)
        throttle = np.random.randint(0, 101)
        brake = np.random.randint(0, 101)
        rpm = np.random.randint(5000, 15000)
        gear = np.random.randint(1, 9)
        drs_enabled = np.random.choice([True, False], p=[0.3, 0.7])

        tire_compound = np.random.choice(tire_compounds)
        tire_age = np.random.randint(0, 50)
        weather = np.random.choice(weather_conditions)

        delta_to_leader = np.random.uniform(-5.0, 10.0)
        corner_speed = np.random.randint(80, 200)
        g_force_lateral = round(np.random.uniform(-3.5, 3.5), 2)

        text = (
            f"Driver {driver} at {circuit}, Lap {lap}, Sector {sector}: "
            f"Speed {speed} km/h, Throttle {throttle}%, Brake {brake}%, "
            f"RPM {rpm}, Gear {gear}, DRS {'ON' if drs_enabled else 'OFF'}, "
            f"Tire {tire_compound} (age {tire_age} laps), "
            f"Weather {weather}, "
            f"Delta to leader {delta_to_leader:+.2f}s, "
            f"Corner speed {corner_speed} km/h, "
            f"Lateral G {g_force_lateral:+.2f}g"
        )

        documents.append({
            "id": f"doc_{doc_id:05d}",
            "text": text,
            "driver": driver,
            "circuit": circuit,
            "lap": lap,
            "sector": sector,
            "speed": speed,
        })
        doc_id += 1

    return documents


def load_fastf1_telemetry(
    year: int = 2023,
    gp: str = "Bahrain",
    session: str = "R"
) -> Optional[list[dict]]:
    """
    Attempt to load real F1 telemetry from FastF1.
    Falls back to synthetic if unavailable.

    Parameters
    ----------
    year : int
        F1 season year
    gp : str
        Grand Prix name
    session : str
        Session type: "R" (Race), "Q" (Qualifying), "FP1"/"FP2"/"FP3"

    Returns
    -------
    list[dict] or None
        List of telemetry documents if successful, None otherwise
    """
    if not HAS_FASTF1:
        return None

    try:
        logger.info(f"Loading FastF1 telemetry for {year} {gp} {session}...")
        session_obj = fastf1.get_session(year, gp, session)
        session_obj.load()

        documents = []
        doc_id = 0

        # Iterate through drivers and their telemetry
        for driver_code in session_obj.drivers:
            try:
                laps = session_obj.laps.pick_driver(driver_code)

                for lap_idx, lap in laps.iterrows():
                    telemetry = lap.get_telemetry()
                    if telemetry is None or len(telemetry) == 0:
                        continue

                    # Chunk telemetry into smaller segments (e.g., every 5 seconds)
                    chunk_size = max(1, len(telemetry) // 5)
                    for chunk_idx in range(0, len(telemetry), chunk_size):
                        chunk = telemetry.iloc[chunk_idx:chunk_idx+chunk_size]

                        avg_speed = chunk["Speed"].mean() if "Speed" in chunk else 0
                        avg_throttle = chunk["Throttle"].mean() if "Throttle" in chunk else 0
                        avg_brake = chunk["Brake"].mean() if "Brake" in chunk else 0
                        tire_compound = lap.get("Compound", "UNKNOWN")

                        text = (
                            f"Driver {driver_code} at {gp}, Lap {lap['LapNumber']}: "
                            f"Speed avg {avg_speed:.1f} km/h, "
                            f"Throttle {avg_throttle:.0f}%, Brake {avg_brake:.0f}%, "
                            f"Tire {tire_compound}"
                        )

                        documents.append({
                            "id": f"doc_{doc_id:05d}",
                            "text": text,
                            "driver": driver_code,
                            "circuit": gp,
                            "lap": int(lap["LapNumber"]),
                        })
                        doc_id += 1
            except Exception as e:
                logger.debug(f"Error processing driver {driver_code}: {e}")
                continue

        if documents:
            logger.info(f"Loaded {len(documents)} documents from FastF1")
            return documents
        else:
            logger.warning("No documents loaded from FastF1; falling back to synthetic")
            return None

    except Exception as e:
        logger.warning(f"FastF1 load failed ({e}); falling back to synthetic")
        return None


def load_telemetry(
    use_synthetic: bool = True,
    fastf1_year: int = 2023,
    fastf1_gp: str = "Bahrain",
    fastf1_session: str = "R",
    synthetic_n: int = 5000,
    seed: int = 42
) -> list[dict]:
    """
    Load telemetry: try FastF1 first, fall back to synthetic.

    Parameters
    ----------
    use_synthetic : bool
        If True, skip FastF1 and use synthetic immediately
    fastf1_year, fastf1_gp, fastf1_session : str
        FastF1 session parameters
    synthetic_n : int
        Number of synthetic documents if needed
    seed : int
        Random seed

    Returns
    -------
    list[dict]
        List of telemetry documents
    """
    if not use_synthetic:
        docs = load_fastf1_telemetry(fastf1_year, fastf1_gp, fastf1_session)
        if docs:
            return docs

    logger.info(f"Generating {synthetic_n} synthetic F1 telemetry documents...")
    return generate_synthetic_telemetry(synthetic_n, seed)


# ─────────────────────────────────────────────────────────────────────
# Embedding and indexing
# ─────────────────────────────────────────────────────────────────────

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to unit vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


# ─────────────────────────────────────────────────────────────────────
# Benchmark queries
# ─────────────────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    "fastest pit stop performance",
    "highest top speed in race",
    "soft tires race strategy",
    "sector 2 performance analysis",
    "driver lap time comparison",
    "wet weather grip conditions",
    "DRS activation patterns",
    "tire degradation analysis",
    "fuel consumption rates",
    "lateral G-force corners",
]


# ─────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────

def benchmark_nanoindex(
    documents: list[dict],
    model: SentenceTransformer,
    seed: int = 42,
) -> dict:
    """
    Benchmark NanoIndex pipeline.

    Parameters
    ----------
    documents : list[dict]
        Telemetry documents
    model : SentenceTransformer
        Embedding model
    seed : int
        Random seed

    Returns
    -------
    dict
        Benchmark results
    """
    if not HAS_NANOINDEX:
        raise RuntimeError(
            "NanoIndex not available. "
            "Install with: pip install nanoindex-rag"
        )

    np.random.seed(seed)

    logger.info("=" * 70)
    logger.info("NANOINDEX PIPELINE")
    logger.info("=" * 70)

    # ─── Embed documents ───
    logger.info(f"\n[1/4] Embedding {len(documents)} documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = normalize_embeddings(embeddings.astype(np.float32))
    logger.info(f"      Embeddings shape: {embeddings.shape}")

    # ─── Build NanoIndex ───
    logger.info("\n[2/4] Building NanoIndex (bits=4)...")
    tracemalloc.start()
    start_time = time.time()

    dim = embeddings.shape[1]
    index = NanoIndex(dim=dim, bits=4, qjl_m=64, seed=seed)

    # Prepare metadata with required "id" and "text" keys
    metadata = [
        {
            "id": doc["id"],
            "text": doc["text"],
            "driver": doc.get("driver"),
            "circuit": doc.get("circuit"),
            "lap": doc.get("lap"),
        }
        for doc in documents
    ]

    index.add(embeddings, metadata)

    build_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = index.stats()
    index_size_mb = stats.get("memory_mb", 0)
    compression_ratio = stats.get("compression_ratio", 0)

    logger.info(f"      Build time: {build_time:.2f}s")
    logger.info(f"      Index size: {index_size_mb:.1f} MB")
    logger.info(f"      Compression ratio: {compression_ratio:.1f}×")
    logger.info(f"      Peak memory: {peak / 1024 / 1024:.1f} MB")

    # ─── Run benchmark queries ───
    logger.info("\n[3/4] Embedding benchmark queries...")
    query_embeddings = model.encode(
        BENCHMARK_QUERIES,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    query_embeddings = normalize_embeddings(query_embeddings.astype(np.float32))
    logger.info(f"      {len(BENCHMARK_QUERIES)} queries embedded")

    # ─── Search queries ───
    logger.info("\n[4/4] Running benchmark queries...")
    latencies = []

    for query_num, (query_text, query_vec) in enumerate(
        zip(BENCHMARK_QUERIES, query_embeddings), 1
    ):
        start_search = time.time()
        results = index.search(query_vec, k=10)
        search_time = (time.time() - start_search) * 1000  # ms
        latencies.append(search_time)

        logger.info(
            f"  Query {query_num}: '{query_text[:40]}...' -> {search_time:.2f}ms"
        )

    # ─── Compute metrics ───
    latencies = np.array(latencies)
    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)

    results = {
        "pipeline": "nanoindex",
        "n_documents": len(documents),
        "index_size_mb": index_size_mb,
        "build_time_s": build_time,
        "query_latency_p50_ms": p50_latency,
        "query_latency_p99_ms": p99_latency,
        "memory_mb": peak / 1024 / 1024,
        "compression_ratio": compression_ratio,
    }

    return results, index


def compare_pipelines(baseline_results: dict, nanoindex_results: dict) -> None:
    """
    Print comparison between baseline and NanoIndex.

    Parameters
    ----------
    baseline_results : dict
        Baseline benchmark results
    nanoindex_results : dict
        NanoIndex benchmark results
    """
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Index size: {baseline_results['index_size_mb']:.1f} MB")
    logger.info(f"  Build time: {baseline_results['build_time_s']:.2f}s")
    logger.info(f"  Query latency (p50): {baseline_results['query_latency_p50_ms']:.2f}ms")
    logger.info(f"  Query latency (p99): {baseline_results['query_latency_p99_ms']:.2f}ms")
    logger.info(f"  Memory: {baseline_results['memory_mb']:.0f} MB")
    logger.info("=" * 70)

    logger.info("\nNANOINDEX PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Index size: {nanoindex_results['index_size_mb']:.1f} MB")
    logger.info(f"  Build time: {nanoindex_results['build_time_s']:.2f}s")
    logger.info(f"  Query latency (p50): {nanoindex_results['query_latency_p50_ms']:.2f}ms")
    logger.info(f"  Query latency (p99): {nanoindex_results['query_latency_p99_ms']:.2f}ms")
    logger.info(f"  Memory: {nanoindex_results['memory_mb']:.0f} MB")
    logger.info(f"  Compression ratio: {nanoindex_results['compression_ratio']:.1f}×")
    logger.info("=" * 70)

    # Compute deltas
    logger.info("\nDELTA (NanoIndex vs Baseline)")
    logger.info("=" * 70)

    size_reduction_pct = (
        100 * (1 - nanoindex_results['index_size_mb'] / baseline_results['index_size_mb'])
    )
    logger.info(f"  Size reduction: {size_reduction_pct:.1f}%")

    latency_overhead_pct = (
        100 * (nanoindex_results['query_latency_p50_ms'] / baseline_results['query_latency_p50_ms'] - 1)
    )
    logger.info(f"  Latency overhead (p50): {latency_overhead_pct:+.1f}%")

    logger.info("=" * 70)


def main():
    """Main entry point."""
    try:
        logger.info("\nF1 Telemetry RAG Benchmark (NanoIndex + Baseline Comparison)")
        logger.info("=" * 70)

        # ─── Load telemetry ───
        documents = load_telemetry(use_synthetic=True, synthetic_n=5000, seed=42)
        logger.info(f"\nLoaded {len(documents)} telemetry documents")

        # ─── Load embedding model ───
        logger.info("\nLoading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # ─── Benchmark baseline (in-memory numpy) ───
        from prod_pipeline_baseline import benchmark_baseline
        baseline_results, _ = benchmark_baseline(documents, model, seed=42)

        # ─── Benchmark NanoIndex ───
        nanoindex_results, _ = benchmark_nanoindex(documents, model, seed=42)

        # ─── Compare ───
        compare_pipelines(baseline_results, nanoindex_results)

        # ─── Save results ───
        output_path = Path(__file__).parent / "nanoindex_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "baseline": baseline_results,
                "nanoindex": nanoindex_results,
            }, f, indent=2)
        logger.info(f"\nDetailed results saved to {output_path}")

        return baseline_results, nanoindex_results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    import traceback
    main()
