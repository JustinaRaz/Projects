from fastapi import Request

from backend.settings.client import WeaviateClient


def get_weaviate(request: Request) -> WeaviateClient:
    client = request.app.state.client
    return client