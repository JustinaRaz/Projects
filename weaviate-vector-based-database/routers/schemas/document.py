from pydantic import BaseModel


class DocumentUploadedResponse(BaseModel):
    """Response returned after a document has been successfully uploaded."""

    filename: str
    content_type: str
    size_bytes: int
    collection_name: str