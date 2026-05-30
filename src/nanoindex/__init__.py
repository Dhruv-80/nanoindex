"""
NanoIndex — compressed vector index for any RAG pipeline.

Based on PolarQuant + QJL (TurboQuant, Google Research, ICLR 2026).
Achieves 7-10× storage reduction with no training required.

    from nanoindex import NanoIndex

    idx = NanoIndex(dim=384, bits=4)
    idx.add(embeddings, metadata)
    results = idx.search(query, k=10)
"""

from .filters import apply_filters
from .index import NanoIndex, SearchResult

__all__ = ["NanoIndex", "SearchResult", "apply_filters"]
__version__ = "0.1.1"
