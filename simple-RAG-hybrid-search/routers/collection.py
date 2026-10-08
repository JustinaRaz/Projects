import logging

from fastapi import APIRouter, Depends, HTTPException
from weaviate.exceptions import WeaviateBaseError

from routers.schemas.collection import (
    CollectionCreatedResponse,
    CollectionDeletedResponse,
    CollectionsListResponse,
    CreateCollectionRequest,
)
from src.core.api import get_weaviate
from src.core.weaviate_client import WeaviateClient

logger = logging.getLogger("app")

router = APIRouter(prefix="/collections", tags=["Collections"])

@router.post("/{collection_name}",
             status_code = 201,
             response_model = CollectionCreatedResponse,
             summary = "Create a collection",
             description = "Creates a new collection in the Weaviate database.")

def create_collection(collection_name: str,
                      payload: CreateCollectionRequest,
                      client: WeaviateClient = Depends(get_weaviate) # noqa: B008
                      ) -> CollectionCreatedResponse:

    if client.collection_exists(collection_name):
        error_message = f"Collection '{collection_name}' already exists."
        raise HTTPException(
            status_code=409,
            detail=error_message
        )

    try:
        client.create_collection(collection_name, description = payload.description)
        logger.info("Collection '%s' created successfully.", collection_name)

    except WeaviateBaseError as exc:
        error_message = f"Failed to create collection '{collection_name}'"
        logger.exception(error_message)
        raise HTTPException(
            status_code=502,
            detail=error_message,
        ) from exc

    return CollectionCreatedResponse(collection_created=collection_name)

@router.get("/inspect_collections",
            response_model=CollectionsListResponse,
            summary = "List collections",
            description = "List the overview of the available collections.")
def get_collections(client: WeaviateClient = Depends(get_weaviate)): # noqa: B008
    try:
        collections = client.list_collections()
        logger.info(f"Overview of available collections: {collections}")

    except WeaviateBaseError as exc:
        error_message = "Failed to list collections."
        logger.exception(error_message)
        raise HTTPException(status_code=502,
                            detail=error_message
                            ) from exc

    return CollectionsListResponse(collections=list(collections))

@router.get("/inspect_collection",
            summary = "Inspect a collection",
            description = "Inspect a specific collection.")
def inspect_collection(collection_name: str,
                       client: WeaviateClient = Depends(get_weaviate)): # noqa: B008

    collection = client.inspect_collection(collection_name)

    return collection

@router.delete("/{collection_name}",
               response_model=CollectionDeletedResponse,
               summary = "Delete collection",
               description = "Removes the collection and its contents from cluster.")
def delete_collections(collection_name: str,
                       client: WeaviateClient = Depends(get_weaviate) # noqa: B008
                       ) -> CollectionDeletedResponse:

    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist."
        )

    try:
        client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' was deleted.")

    except WeaviateBaseError as exc:
        error_message = f"Failed to delete collection '{collection_name}'."
        logger.exception(error_message)
        raise HTTPException(status_code=502, detail=error_message) from exc

    return CollectionDeletedResponse(collection_deleted=collection_name)
