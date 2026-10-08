import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from routers.schemas.document import DocumentUploadedResponse
from src.core.api import get_weaviate
from src.core.weaviate_client import WeaviateClient
from src.processing.preprocessor import extract_document
from src.rag.llm import Gemma

logger = logging.getLogger("app")


router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post(
    "/get_document_text",
    summary="Extract the text from a document.",
    description="Upload the document and retrieve the text in a Markdown file."
)
async def get_document(
    file: UploadFile = File(...)  # noqa: B008
    ):

    contents = await file.read()
    extract_document(contents=contents, filename=file.filename)

    return Response(
        media_type = "text/markdown",
    )

@router.post(
    "/upload",
    status_code=201,
    response_model=DocumentUploadedResponse,
    summary="Upload a document",
    description="Uploads, chunks, embeds, and stores document in a collection.",
)
async def upload_document(
    collection_name: str = Form(...),
    file: UploadFile = File(...),  # noqa: B008
    client: WeaviateClient = Depends(get_weaviate),  # noqa: B008
) -> DocumentUploadedResponse:
    if file.filename is None or file.content_type is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not contain a valid filename/content type.",
        )

    contents = await file.read()

    try:
        logger.info(f"The upload of {file.filename} is starting.")
        result = client.index_document(
            contents=contents,
            filename=file.filename,
            collection_name=collection_name,
        )
        logger.info("Document %s was uploaded successfully.", file.filename)
    except Exception as exc:
        logger.exception("Failed to parse '%s'", file.filename)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while parsing the document.",
        ) from exc

    return DocumentUploadedResponse(
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(contents),
        collection_name=collection_name,
        chunks_indexed=result.chunks_indexed)

@router.get(
    "/hybrid_search",
    summary="Perform hybrid search",
    description="Hybrid search using Weaviate's BM25 and all-MiniLM-L6-v2 dense vectors"
)

def perform_hybrid_search(question: str,
                          client: WeaviateClient = Depends(get_weaviate), # noqa: B008
                          collection_name: str = "Testing"):

    response = client.hybrid_search(collection_name = collection_name,
                                    query = question)
    return response


@router.get(
    "/llm_retrieval",
    summary="Perform hybrid search + LLM request",
    description="Performs the search of information, sends retrieved chunks to LLM.")

def get_answer(question: str,
               client: WeaviateClient = Depends(get_weaviate), # noqa: B008
               collection_name: str = "Testing",
    ):

    chunks = perform_hybrid_search(question=question,
                                   client = client,
                                   collection_name = collection_name)
    gemma = Gemma()
    response = gemma.generate(question=question,
                              context=chunks)

    return response
