from fastapi import APIRouter, Depends, HTTPException

from backend.settings.client import WeaviateClient
from dependency import get_weaviate

router = APIRouter(
    tags=["Documents"]
)


@router.post("/create_collection",
             summary = "Create a collection",
             description = "Creates a new collection in the Weaviate database.")