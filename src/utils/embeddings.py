"""
Embedding utility – wraps sentence-transformers for CPU-only embeddings.
Keeps VRAM free for Ollama.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Lightweight wrapper around SentenceTransformer."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        device: str = config.EMBEDDING_DEVICE,
    ):
        logger.info("Loading embedding model %s on %s …", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = config.EMBEDDING_DIMENSION

    def embed_text(self, text: str) -> List[float]:
        """Embed a single piece of text → list of floats."""
        return self.model.encode(text, show_progress_bar=False).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a batch of texts → list of float-lists."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        return embeddings.tolist()


# Module-level singleton (lazy)
_engine = None


def get_embedding_engine() -> EmbeddingEngine:
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine


def embed_text(text: str) -> List[float]:
    return get_embedding_engine().embed_text(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    return get_embedding_engine().embed_batch(texts)
