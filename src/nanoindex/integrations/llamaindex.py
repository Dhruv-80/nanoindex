"""
LlamaIndex VectorStore integration for NanoIndex.

Usage
-----
from nanoindex.integrations.llamaindex import NanoVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

store   = NanoVectorStore(dim=384, bits=4)
storage = StorageContext.from_defaults(vector_store=store)
index   = VectorStoreIndex.from_documents(documents, storage_context=storage)

results = index.as_query_engine().query("your question")
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError as e:
    raise ImportError(
        "LlamaIndex is required for this integration. "
        "Install it with: pip install llama-index-core"
    ) from e

from nanoindex import NanoIndex


class NanoVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex VectorStore backed by NanoIndex.

    Reduces embedding storage by 7-10× with no training required.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    _index: NanoIndex

    def __init__(self, dim: int = 384, bits: int = 4, qjl_m: int = 64, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_index", NanoIndex(dim=dim, bits=bits, qjl_m=qjl_m))

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    @classmethod
    def class_name(cls) -> str:
        return "NanoVectorStore"

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        vecs = np.array(
            [node.get_embedding() for node in nodes], dtype=np.float32
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.clip(norms, 1e-8, None)

        records = [
            {
                "id":   node.node_id,
                "text": node.get_content(),
                **node.metadata,
            }
            for node in nodes
        ]

        self._index.add(vecs, records)
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        raise NotImplementedError("NanoIndex does not support deletion.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        q_vec = np.array(query.query_embedding, dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec) + 1e-8

        filters = _convert_filters(query.filters) if query.filters else None
        results = self._index.search(q_vec, k=query.similarity_top_k, filters=filters)

        nodes  = []
        scores = []
        ids    = []
        for r in results:
            node = TextNode(
                text=r.text,
                id_=r.id,
                metadata=r.metadata,
            )
            nodes.append(node)
            scores.append(r.score)
            ids.append(r.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, persist_path: str, **kwargs: Any) -> None:
        self._index.save(persist_path)

    @classmethod
    def from_persist_path(cls, persist_path: str, **kwargs) -> "NanoVectorStore":
        store = cls.__new__(cls)
        object.__setattr__(store, "_index", NanoIndex.load(persist_path))
        return store


def _convert_filters(filters: MetadataFilters) -> dict:
    """Convert LlamaIndex MetadataFilters to NanoIndex filter dict."""
    result: dict = {}
    for f in filters.filters:
        result[f.key] = f.value
    return result
