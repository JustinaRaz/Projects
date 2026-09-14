from fastapi import APIRouter, Depends, HTTPException

from backend.settings.client import WeaviateClient
from dependency import get_weaviate

router = APIRouter(
    tags=["Collections"]
)


@router.post("/create_collection",
             summary = "Create a collection",
             description = "Creates a new collection in the Weaviate database.")

def create_collection(collection_name: str,
                      description: str | None = None,
                      client: WeaviateClient = Depends(get_weaviate)):  # noqa: B008
    
    if client.collection_exists(collection_name):
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{collection_name}' already exists. Please use another collection name."
        )

    client.create_collection(collection_name, description = description)

    return {"collection_created": collection_name}

@router.get("/list_collections",
             summary = "Get an overview of collections",
             description = "List the overview of the available collections on a Weaviate cluster.")

def get_collections(client: WeaviateClient = Depends(get_weaviate)): # noqa: B008
    
    collections = client.list_collections()

    return {"collections": collections}

@router.delete("/delete_collection",
             summary = "Delete collection",
             description = "Removes the collection and its contents from cluster.")

def delete_collections(collection_name: str,
                       client: WeaviateClient = Depends(get_weaviate)): # noqa: B008
    
    if not client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' does not exist."
        )

    client.delete_collection(collection_name)

    return {"collection_deleted": collection_name}