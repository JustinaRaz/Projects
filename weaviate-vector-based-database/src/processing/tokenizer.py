import logging
from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.core.settings import get_settings

logger = logging.getLogger("src")


@lru_cache
def get_tokenizer() -> PreTrainedTokenizerBase:
    """Return the tokenizer matching the configured embedding model.

    Tokenizer is loaded once per process.

    Returns:
        A HuggingFace tokenizer whose vocabulary and token limits match
        the embedding model, so chunks are sized against the model's
        actual constraints rather than an approximation.
    """
    settings = get_settings()
    logger.debug("Loading tokenizer for '%s'", settings.embedding_model_name)
    return AutoTokenizer.from_pretrained(settings.embedding_model_name)
