"""Query-time retrieval of relevant chunks from a Weaviate collection."""

import logging

from src.core.weaviate_client import WeaviateClient
from src.rag.embedder import get_embedder

logger = logging.getLogger("app")


def retrieve(
    client: WeaviateClient,
    collection_name: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Retrieve the chunks most relevant to a query.

    Args:
        client: Active WeaviateClient instance.
        collection_name: Collection to search.
        query: Natural-language query.
        limit: Max chunks to return.

    Returns:
        Matching chunk property dicts, ordered by relevance.
    """
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)
    results = client.search(collection_name, query_vector, limit=limit)
    logger.debug("Retrieved %d chunks for query in '%s'", len(results), collection_name)
    return results