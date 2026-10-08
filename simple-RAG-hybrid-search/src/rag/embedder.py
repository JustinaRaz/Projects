import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.core.settings import get_settings

logger = logging.getLogger("src")
settings = get_settings()

class Embedder:
    """Wraps an embedding model, converting text into vectors for Weaviate."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        logger.info("Loading embedding model '%s'", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (e.g. document chunks).

        Args:
            texts: A list of texts to embed.

        Returns:
            One embedding vector per input text, same order.
        """
        if not texts:
            return []

        embeddings = self._model.encode(texts,
                                        convert_to_numpy=True,
                                        show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string, for retrieval."""
        return self.embed_texts([query])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Return the process-wide cached Embedder instance.
    """
    return Embedder()
