from fastapi import Request

from src.core.weaviate_client import WeaviateClient


def get_weaviate(request: Request) -> WeaviateClient:
    """Return the shared WeaviateClient instance for dependency injection.

    The client is created once at startup and stored on app.state.client.
    Retrieving it here avoids reconnection on every call.

    Args:
        request: The current request, used to access app.state.

    Returns:
        The application's shared WeaviateClient instance.
    """
    
    client = request.app.state.client
    return client