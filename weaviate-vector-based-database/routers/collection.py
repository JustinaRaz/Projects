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
                      client: WeaviateClient = Depends(get_weaviate)) -> CollectionCreatedResponse:  # noqa: B008
    
    if client.collection_exists(collection_name):
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{collection_name}' already exists. Please use another collection name."
        )
    
    try:
        client.create_collection(collection_name, description = payload.description)
    except WeaviateBaseError:
        logger.exception("Failed to create collection '%s'", collection_name)
        raise HTTPException(status_code=502, detail="Failed to create collection.")

    logger.info("Collection created: %s", collection_name)
    return CollectionCreatedResponse(collection_created=collection_name)

@router.get("",
            response_model=CollectionsListResponse,
             summary = "Get an overview of collections",
             description = "List the overview of the available collections on a Weaviate cluster.")
def get_collections(client: WeaviateClient = Depends(get_weaviate)): # noqa: B008
    try:
        collections = client.list_collections()
    except WeaviateBaseError:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=502, detail="Failed to retrieve collections.")

    return CollectionsListResponse(collections=list(collections))

@router.delete("/{collection_name}",
               response_model=CollectionDeletedResponse,
               summary = "Delete collection",
               description = "Removes the collection and its contents from cluster.")
def delete_collections(collection_name: str,
                       client: WeaviateClient = Depends(get_weaviate)) -> CollectionDeletedResponse: # noqa: B008
    
    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist."
        )

    try:
        client.delete_collection(collection_name)
    except WeaviateBaseError:
        logger.exception("Failed to delete collection '%s'", collection_name)
        raise HTTPException(status_code=502, detail="Failed to delete collection.")

    logger.info("Collection deleted: %s", collection_name)
    return CollectionDeletedResponse(collection_deleted=collection_name)