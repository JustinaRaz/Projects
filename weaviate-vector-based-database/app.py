from fastapi import FastAPI

from backend.routers import collection
from backend.settings.client import WeaviateClient


def create_app() -> FastAPI:
    app = FastAPI(
        title="Practice Project",
        description="Created for practicing reasons.",
    )

    weaviate_client = WeaviateClient()
    app.state.client = weaviate_client

    app.include_router(collection.router)

    return app


app = create_app()