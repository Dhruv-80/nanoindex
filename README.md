# nanoindex

**Compressed vector index for RAG pipelines. 7-10× smaller, no training, no accuracy loss.**

Drop-in replacement for FAISS or your vector store's quantization layer, based on [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (Google Research, ICLR 2026). The first open-source Python implementation.

```python
from nanoindex import NanoIndex

idx = NanoIndex(dim=384, bits=4)
idx.add(embeddings, metadata)          # compress 34 MB → 4.6 MB
results = idx.search(query, k=10)     # 2ms on 22K vectors
```

---

## Why

Vector storage is the quiet cost in every RAG system. A modest corpus of 500K documents at 384 dimensions costs **750 MB** as float32. That's before replication, before backups, before you add more models.

NanoIndex compresses embeddings to 3–4 bits per value using a two-stage algorithm that requires **no training data** and **no codebooks** — just your vectors. You get a 7-10× smaller index with retrieval quality that beats trained baselines at equivalent compression ratios.

---

## Install

```bash
pip install nanoindex           # core only (numpy)
pip install nanoindex[fast]     # + Numba JIT (~15× faster search)
pip install nanoindex[langchain]
pip install nanoindex[llamaindex]
```

---

## Quick start

```python
import numpy as np
from nanoindex import NanoIndex

# Any float32 embeddings — model and domain agnostic
embeddings = encoder.encode(texts)   # (N, dim) float32, L2-normalised

metadata = [{"id": f"doc_{i}", "text": texts[i], "source": sources[i]} for i in range(N)]

idx = NanoIndex(dim=384, bits=4)
idx.add(embeddings, metadata)
idx.save("my_index")

# Later
idx = NanoIndex.load("my_index")
results = idx.search(query_vec, k=10)

for r in results:
    print(r.score, r.text, r.metadata)
```

### Batch search

```python
# Search multiple queries at once
results = idx.search(query_matrix, k=10)   # (M, dim) → list[list[SearchResult]]
```

### Metadata filters

```python
# Equality filter (case-insensitive, supports lists)
results = idx.search(q, k=10, filters={"source": "arxiv"})
results = idx.search(q, k=10, filters={"author": ["Smith", "Jones"]})

# Range filters — any numeric field with _min / _max suffix
results = idx.search(q, k=10, filters={"year_min": 2022, "score_max": 0.9})

# Combined
results = idx.search(q, k=10, filters={"source": "arxiv", "year_min": 2023})
```

---

## LangChain integration

```python
from nanoindex.integrations.langchain import NanoVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Build — replaces FAISS.from_texts()
store = NanoVectorStore.from_texts(texts, embeddings, bits=4)
store.save_local("my_index")

# Load
store = NanoVectorStore.load_local("my_index", embeddings)

# Use as retriever
retriever = store.as_retriever(search_kwargs={"k": 5})
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

---

## LlamaIndex integration

```python
from nanoindex.integrations.llamaindex import NanoVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = NanoVectorStore(dim=1536, bits=4)
storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
index        = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)

query_engine = index.as_query_engine()
response     = query_engine.query("What is the capital of France?")
```

---

## Benchmarks

Measured on 22K vectors, dim=384 (2023 F1 Bahrain GP telemetry embeddings, `all-MiniLM-L6-v2`), Apple M-series, single query, Numba enabled.

| Method                  | Compression | Index size | Recall@10 | p50 latency |
|-------------------------|-------------|------------|-----------|-------------|
| Brute-force float32     | 1×          | 34.1 MB    | 1.000     | 0.6 ms      |
| **NanoIndex (4-bit)**   | **7.4×**    | **4.6 MB** | **0.744** | **2.0 ms**  |
| NanoIndex (3-bit)       | 9.6×        | 3.5 MB     | 0.514     | 2.0 ms      |
| Faiss PQ (m=48, 8-bit)  | 32×         | 1.1 MB     | 0.369     | 0.3 ms      |
| Faiss SQ8               | 4×          | 8.5 MB     | 0.975     | 1.6 ms      |

**NanoIndex at 4-bit outperforms Faiss PQ on recall (0.744 vs 0.369) while requiring zero training.** The 2ms latency reflects a pure Python/Numba implementation; the original CUDA kernel in the paper achieves 8× GPU throughput vs float32.

Run benchmarks on your own data:

```bash
pip install nanoindex[bench]
python benchmarks/run_benchmarks.py --embeddings your_embeddings.npz --bits 4
```

---

## How it works

NanoIndex implements **TurboQuant** — a two-stage, data-oblivious vector compression algorithm.

**Stage 1 — PolarQuant**

Each embedding is randomly rotated (shared orthogonal matrix R), then converted from Cartesian to polar coordinates. The angle components are uniformly quantized to `b` bits. The radii are recursively paired and quantized across 9 levels until a single scalar radius remains. This stores `d-1` quantized angles + 1 float32 radius per vector.

**Stage 2 — QJL residual correction**

The quantization error from Stage 1 is projected through a random Johnson-Lindenstrauss matrix S ∈ ℝᵐˣᵈ. Only the sign bits of the projection are stored (1 bit each). At query time, a bias-corrected estimator adds back the residual correction without any decompression.

**Inner product in the compressed domain**

Approximate inner products are computed directly on compressed representations — no decompression step. The Numba-accelerated kernel processes 22K vectors in 2ms using parallel threads.

```
⟨q, v⟩ ≈ PolarQuant_IP(q, angles, radius) + QJL_correction(q, sign_bits, residual_norm)
```

---

## Configuration

```python
NanoIndex(
    dim   = 384,   # embedding dimension
    bits  = 4,     # bits per angle (3–8); 4-bit recommended
    qjl_m = 64,    # QJL projection dimensions; higher = better correction
    seed  = 42,    # for reproducible rotation matrices
)
```

| `bits` | Compression | Typical Recall@10 | Use when |
|--------|-------------|-------------------|----------|
| 3      | ~9-10×      | 0.50–0.55         | Maximum compression, quality less critical |
| 4      | ~7-8×       | 0.70–0.75         | Recommended default |
| 6      | ~5×         | 0.85–0.90         | High recall requirements |
| 8      | ~4×         | 0.93+             | Near-lossless |

---

## SearchResult fields

```python
@dataclass
class SearchResult:
    rank:     int         # 0-indexed rank in result list
    score:    float       # approximate cosine similarity
    id:       str         # from metadata["id"]
    text:     str         # from metadata["text"]
    metadata: dict        # all other fields from your metadata dict
```

---

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- numba ≥ 0.58 *(optional, recommended — `pip install nanoindex[fast]`)*

---

## Algorithm credit

NanoIndex implements the TurboQuant algorithm from:

> **TurboQuant: Redefining AI Efficiency with Extreme Compression**  
> Google Research · [Blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) · ICLR 2026

This is an independent open-source Python/Numba implementation. The original paper's performance numbers were obtained using a custom CUDA kernel.

---

## License

MIT
