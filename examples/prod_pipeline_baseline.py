#!/usr/bin/env python3
"""
Production-grade baseline RAG pipeline for F1 telemetry.

Baseline: in-memory numpy vectors (no compression).
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.
Generates or loads real F1 telemetry, chunks into ~5000 documents.
Measures index size, build time, query latency, and memory usage.

Run with: python prod_pipeline_baseline.py
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

# Try to import FastF1; fall back to synthetic generation if unavailable
try:
    import fastf1
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False
    logger.warning("FastF1 not available; will use synthetic F1 telemetry data")


# ─────────────────────────────────────────────────────────────────────
# Synthetic F1 telemetry data generation
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


class BaselineIndex:
    """Simple in-memory baseline index using numpy arrays."""

    def __init__(self, embeddings: np.ndarray, documents: list[dict]):
        """
        Initialize baseline index.

        Parameters
        ----------
        embeddings : (N, dim) float32
            L2-normalized embeddings
        documents : list[dict]
            Documents with "id" and "text" keys
        """
        self.embeddings = embeddings.astype(np.float32)
        self.documents = documents
        self.dim = embeddings.shape[1]
        self.n_vectors = embeddings.shape[0]

    def search(self, query: np.ndarray, k: int = 10) -> list[dict]:
        """
        Search using cosine similarity (inner product on normalized vectors).

        Parameters
        ----------
        query : (dim,) float32
            L2-normalized query embedding
        k : int
            Number of results to return

        Returns
        -------
        list[dict]
            Top-k results with rank, score, id, text
        """
        # Cosine similarity = inner product on normalized vectors
        scores = self.embeddings @ query

        k_actual = min(k, len(scores))
        top_idx = np.argpartition(scores, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for rank, idx in enumerate(top_idx):
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "id": self.documents[idx]["id"],
                "text": self.documents[idx]["text"],
            })
        return results

    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        # embeddings only (float32 = 4 bytes per value)
        return (self.embeddings.nbytes + self.dim * 4) / (1024 * 1024)


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

def benchmark_baseline(
    documents: list[dict],
    model: SentenceTransformer,
    seed: int = 42,
) -> dict:
    """
    Benchmark baseline pipeline.

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
    np.random.seed(seed)

    logger.info("=" * 70)
    logger.info("BASELINE PIPELINE")
    logger.info("=" * 70)

    # ─── Embed documents ───
    logger.info(f"\n[1/4] Embedding {len(documents)} documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = normalize_embeddings(embeddings.astype(np.float32))
    logger.info(f"      Embeddings shape: {embeddings.shape}")

    # ─── Build index ───
    logger.info("\n[2/4] Building baseline index...")
    tracemalloc.start()
    start_time = time.time()

    index = BaselineIndex(embeddings, documents)

    build_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    index_size_mb = index.memory_mb()
    logger.info(f"      Build time: {build_time:.2f}s")
    logger.info(f"      Index size: {index_size_mb:.1f} MB")
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
        "pipeline": "baseline",
        "n_documents": len(documents),
        "index_size_mb": index_size_mb,
        "build_time_s": build_time,
        "query_latency_p50_ms": p50_latency,
        "query_latency_p99_ms": p99_latency,
        "memory_mb": peak / 1024 / 1024,
        "compression_ratio": 1.0,
    }

    return results, index


def main():
    """Main entry point."""
    try:
        logger.info("\nF1 Telemetry RAG Benchmark (Baseline)")
        logger.info("=" * 70)

        # ─── Load telemetry ───
        documents = load_telemetry(use_synthetic=True, synthetic_n=5000, seed=42)
        logger.info(f"\nLoaded {len(documents)} telemetry documents")

        # ─── Load embedding model ───
        logger.info("\nLoading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # ─── Run benchmark ───
        results, index = benchmark_baseline(documents, model, seed=42)

        # ─── Print results ───
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE RESULTS")
        logger.info("=" * 70)
        logger.info(f"  Index size: {results['index_size_mb']:.1f} MB")
        logger.info(f"  Build time: {results['build_time_s']:.2f}s")
        logger.info(f"  Query latency (p50): {results['query_latency_p50_ms']:.2f}ms")
        logger.info(f"  Query latency (p99): {results['query_latency_p99_ms']:.2f}ms")
        logger.info(f"  Memory: {results['memory_mb']:.0f} MB")
        logger.info("=" * 70)

        # ─── Save results ───
        output_path = Path(__file__).parent / "baseline_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

        return results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    import traceback
    main()
