import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from routers.schemas.document import DocumentUploadedResponse
from src.core.settings import get_settings

settings = get_settings()
logger = logging.getLogger("app")

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post(
    "/upload",
    status_code=201,
    summary="Upload a document",
    description="Uploads a document and associates it with collection.",
)
async def upload_document(
    collection_name: str = Form(...),
    file: UploadFile = File(...),  # noqa: B008
) -> DocumentUploadedResponse:
    if file.filename is None or file.content_type is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is missing a filename or content type.",
        )

    contents = await file.read()

    # ... process/store the document, pass to Weaviate ingestion, etc.

    logger.info(
        "Document uploaded: %s (%s, %d bytes) -> collection '%s'",
        file.filename,
        file.content_type,
        len(contents),
        collection_name,
    )

    return DocumentUploadedResponse(
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(contents),
        collection_name=collection_name,
    )