"""
LangChain VectorStore integration for NanoIndex.

Usage
-----
from nanoindex.integrations.langchain import NanoVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store = NanoVectorStore.from_texts(texts, embeddings, bits=4)
docs  = store.similarity_search("your query", k=5)
store.save_local("my_index")

store = NanoVectorStore.load_local("my_index", embeddings)
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install it with: pip install langchain-core"
    ) from e

from nanoindex import NanoIndex


class NanoVectorStore(VectorStore):
    """
    LangChain-compatible VectorStore backed by NanoIndex.

    Drop-in replacement for FAISS, Chroma, or any other LangChain vector store.
    Reduces embedding storage by 7-10× with no training required.
    """

    def __init__(self, embedding: Embeddings, index: NanoIndex):
        self._embedding = embedding
        self._index     = index

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]

        vecs = np.array(self._embedding.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.clip(norms, 1e-8, None)

        ids = [str(i) for i in range(self._index.n_vectors, self._index.n_vectors + len(texts))]
        records = [
            {"id": id_, "text": text, **meta}
            for id_, text, meta in zip(ids, texts, metadatas)
        ]

        if self._index.n_vectors == 0:
            self._index.add(vecs, records)
        else:
            # Append by re-compressing the full set (incremental add not yet supported)
            raise NotImplementedError(
                "NanoIndex does not yet support incremental add. "
                "Build the full index with from_texts() instead."
            )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        results = self.similarity_search_with_score(query, k=k, filter=filter)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        q_vec = np.array(self._embedding.embed_query(query), dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec) + 1e-8

        results = self._index.search(q_vec, k=k, filters=filter)
        return [
            (Document(page_content=r.text, metadata={"id": r.id, **r.metadata}), r.score)
            for r in results
        ]

    def _select_relevance_score_fn(self):
        return lambda score: score  # scores are cosine similarities in [-1, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_local(self, path: str) -> None:
        self._index.save(path)

    @classmethod
    def load_local(cls, path: str, embedding: Embeddings, **kwargs) -> "NanoVectorStore":
        index = NanoIndex.load(path)
        return cls(embedding=embedding, index=index)

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        bits: int = 4,
        qjl_m: int = 64,
        **kwargs: Any,
    ) -> "NanoVectorStore":
        metadatas = metadatas or [{} for _ in texts]

        vecs = np.array(embedding.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.clip(norms, 1e-8, None)

        dim = vecs.shape[1]
        index = NanoIndex(dim=dim, bits=bits, qjl_m=qjl_m)

        records = [
            {"id": str(i), "text": text, **meta}
            for i, (text, meta) in enumerate(zip(texts, metadatas))
        ]
        index.add(vecs, records)

        return cls(embedding=embedding, index=index)

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: list[tuple[str, list[float]]],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        bits: int = 4,
        qjl_m: int = 64,
        **kwargs: Any,
    ) -> "NanoVectorStore":
        texts = [t for t, _ in text_embeddings]
        vecs  = np.array([e for _, e in text_embeddings], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.clip(norms, 1e-8, None)

        metadatas = metadatas or [{} for _ in texts]
        dim = vecs.shape[1]
        index = NanoIndex(dim=dim, bits=bits, qjl_m=qjl_m)

        records = [
            {"id": str(i), "text": text, **meta}
            for i, (text, meta) in enumerate(zip(texts, metadatas))
        ]
        index.add(vecs, records)

        return cls(embedding=embedding, index=index)
