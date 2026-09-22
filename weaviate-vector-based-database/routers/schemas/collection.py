from pydantic import BaseModel


class CreateCollectionRequest(BaseModel):
    description: str | None = None


class CollectionCreatedResponse(BaseModel):
    collection_created: str


class CollectionsListResponse(BaseModel):
    collections: list[str]


class CollectionDeletedResponse(BaseModel):
    collection_deleted: str