import logging

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth

from routers.schemas.document import IndexResult
from src.core.settings import Settings, get_settings
from src.processing.chunking import get_chunker
from src.processing.preprocessor import extract_document
from src.rag.embedder import get_embedder

logger = logging.getLogger("src")


class WeaviateClient:
    """WeaviateClient: holds all relevant functions when interracting with the database.
    """

    def __init__(self, settings: Settings | None = None):
        """Connects to the Weaviate client once the class is instantiated.
        Takes .env values.
        """
        settings = settings or get_settings()

        self.url = settings.weaviate_url
        self.api_key = settings.weaviate_api_key

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
        )

    def close(self):
        self.client.close()

    def create_collection(self,
                          collection_name: str,
                          description: str | None = None):
        return self.client.collections.create(
            name=collection_name,
            description=description,
            vector_config=Configure.Vectors.self_provided()
        )

    def get_collection(self, collection_name):
        return self.client.collections.use(collection_name)

    def delete_collection(self, collection_name):
        self.client.collections.delete(collection_name)

    def list_collections(self):
        return self.client.collections.list_all()

    def inspect_collection(self, collection_name: str):
        collection = self.get_collection(collection_name)
        response = collection.query.fetch_objects(include_vector=True)

        return response.objects

    def add_objects(
        self,
        collection_name: str,
        properties: list[dict],
        vectors: list[list[float]],
    ) -> int:

        if len(properties) != len(vectors):
            error_message = "The number of properties must match the number of vectors."
            raise ValueError(error_message)

        collection = self.get_collection(collection_name)

        for object_properties, vector in zip(properties, vectors, strict=True):
            collection.data.insert(
                properties=object_properties,
                vector=vector,
            )

        return len(properties)

    def store_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> int:
        """Store embedded chunks in a Weaviate collection."""

        self.add_objects(
            collection_name,
            chunks,
            vectors,
        )

        logger.info(
            "Stored %d chunks in collection '%s'",
            len(chunks),
            collection_name,
        )

        return len(chunks)

    def index_document(
        self,
        contents: bytes,
        collection_name: str,
        filename: str
    ) -> IndexResult:

        """Runs the full ingestion pipeline for one uploaded document.

        Steps:
            1. Parse (using docling)
            2. Chunk semantically
            3. Embed each chunk
            4. Store chunks + vectors in the target Weaviate collection

        Args:
            contents: Raw bytes of the uploaded file.
            filename: Original filename.
            collection_name: Target Weaviate collection.

        Returns:
            A summary of what was indexed.
            """

        document = extract_document(contents, filename)
        chunks = get_chunker().chunk_document(
                document,
                source_filename=filename,
            )
        vectors = get_embedder().embed_texts([chunk.text for chunk in chunks])

        properties = [{"content": chunk.text, **chunk.metadata} for chunk in chunks]
        stored = self.store_chunks(collection_name, properties, vectors)

        return IndexResult(
            filename=filename,
            collection_name=collection_name,
            chunks_indexed=stored,
        )

    def send_embeddings(
        self,
        collection_name: str,
        text_chunks: list[dict],
        vectors: list[list[float]],
    ):
        """Store embedded chunks in a Weaviate collection."""

        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)

        self.add_objects(
            collection_name,
            text_chunks,
            vectors,
        )

        logger.info(
            "Stored %d chunks in collection '%s'",
            len(text_chunks),
            collection_name,
        )

    def hybrid_search(self, collection_name: str, query: str, alpha = 0.6):
        collection = self.client.collections.use(collection_name)
        query_vector = get_embedder().embed_query(query)
        response = collection.query.hybrid(query=query,
                                            vector=query_vector,
                                            alpha=alpha)
        return response
