"""Wrap sentence-transformers to embed lists of TelemetryChunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..chunking.schemas import TelemetryChunk

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (dim=%d) on %s", model_name, self.dim, device)

    def embed_chunks(self, chunks: list[TelemetryChunk], batch_size: int = 256) -> np.ndarray:
        """Return float32 array of shape (N, dim)."""
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
