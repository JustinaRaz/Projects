from dataclasses import field

from pydantic import BaseModel


class Chunk(BaseModel):
    """A single semantically coherent piece of a document, ready to be embedded."""

    text: str
    chunk_index: int
    chunk_id: str
    metadata: dict = field(default_factory=dict)

class IndexResult(BaseModel):
    """Summary of a single document ingestion run."""

    filename: str
    collection_name: str
    chunks_indexed: int


class DocumentUploadedResponse(BaseModel):
    """Summary of a single document after completed upload to Weaviate collection."""

    filename: str
    content_type: str
    size_bytes: int
    collection_name: str
    chunks_indexed: int
