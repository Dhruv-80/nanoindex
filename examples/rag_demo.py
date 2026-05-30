#!/usr/bin/env python3
"""
Self-contained RAG pipeline demo using sentence-transformers and nanoindex-rag.
Run with: python rag_demo.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ── NANOINDEX-RAG ──────────────────────────────────────────────────
# pip install nanoindex-rag
from nanoindex import NanoIndex, SearchResult
# ───────────────────────────────────────────────────────────────────


# Corpus of 20 AI/ML documents
CORPUS = [
    "Transformers are neural network architectures based on self-attention mechanisms.",
    "RAG stands for Retrieval-Augmented Generation, combining retrieval with generation models.",
    "Vector databases store embeddings for fast similarity search and retrieval.",
    "Large Language Models like GPT-4 are trained on massive text corpora.",
    "Embeddings are dense vector representations of text that capture semantic meaning.",
    "Attention mechanisms allow models to focus on relevant parts of input sequences.",
    "Fine-tuning adapts pre-trained models to specific downstream tasks.",
    "BERT is a bidirectional transformer model trained with masked language modeling.",
    "Semantic search uses embeddings to find similar documents without keyword matching.",
    "Knowledge graphs represent entities and relationships as structured networks.",
    "Prompt engineering optimizes input queries to improve LLM output quality.",
    "Token-level embeddings vs sentence embeddings have different use cases.",
    "Few-shot learning enables models to learn from minimal examples.",
    "Similarity measures like cosine distance rank documents by relevance.",
    "Retrieval-based QA systems fetch relevant documents before answering questions.",
    "Neural networks learn hierarchical representations through multiple layers.",
    "Transformer optimization techniques include knowledge distillation and quantization.",
    "Text preprocessing includes tokenization, stemming, and normalization.",
    "Cross-attention mechanisms enable information flow between different modalities.",
    "Benchmark datasets like SQuAD evaluate question answering systems.",
]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings."""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def build_index(embeddings: np.ndarray, corpus: list[str]) -> NanoIndex:
    """Build NanoIndex from embeddings and corpus."""
    dim = embeddings.shape[1]
    index = NanoIndex(dim=dim, bits=4)

    metadata = [{"id": str(i), "text": text} for i, text in enumerate(corpus)]
    index.add(embeddings, metadata)

    return index


def search_index(index: NanoIndex, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """Search index and return top-k results."""
    results = index.search(query_embedding, k=top_k)

    return [
        {"rank": r.rank + 1, "score": r.score, "text": r.text, "id": r.id}
        for r in results
    ]


def main():
    """Main RAG demo."""
    print("=" * 70)
    print("RAG Pipeline Demo: sentence-transformers + nanoindex-rag")
    print("=" * 70)

    # Load embedding model
    print("\n[1/4] Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed corpus
    print(f"[2/4] Embedding {len(CORPUS)} documents...")
    corpus_embeddings = model.encode(CORPUS, convert_to_numpy=True)
    corpus_embeddings = normalize_embeddings(corpus_embeddings)
    print(f"      Embedding shape: {corpus_embeddings.shape}")

    # Build index
    print("[3/4] Building NanoIndex...")
    index = build_index(corpus_embeddings, CORPUS)
    print("      Index built successfully")

    # Run example queries
    print("[4/4] Running example queries...\n")

    queries = [
        "What are transformers and self-attention?",
        "How do embeddings work for semantic search?",
        "Tell me about retrieval-augmented generation",
        "What is BERT and masked language modeling?",
        "How do benchmark datasets evaluate NLP systems?"
    ]

    for query_num, query in enumerate(queries, 1):
        print(f"\n[Query {query_num}] {query}")
        print("-" * 70)

        # Embed and normalize query
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = normalize_embeddings(query_embedding)
        query_embedding = query_embedding[0]  # Get single embedding

        # Search
        results = search_index(index, query_embedding, top_k=3)

        for result in results:
            print(f"  [{result['rank']}] Score: {result['score']:.4f}")
            print(f"      {result['text'][:60]}...")


if __name__ == "__main__":
    main()
