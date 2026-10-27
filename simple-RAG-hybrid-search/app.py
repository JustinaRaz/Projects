import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers import collection, document
from src.core.logger_config import setup_logging
from src.core.settings import get_settings
from src.core.weaviate_client import WeaviateClient

settings = get_settings()
settings.init_dirs()

setup_logging()
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the Weaviate client lifecycle."""
    logger.info("Application starting")

    client = WeaviateClient()
    app.state.client = client

    try:
        yield
    finally:
        logger.info("Closing Weaviate client")
        client.close()
        logger.info("Weaviate client closed")


app = FastAPI(
    title="Practice Project",
    description="Created for practicing reasons.",
    lifespan=lifespan,
)

app.include_router(collection.router)
app.include_router(document.router)
