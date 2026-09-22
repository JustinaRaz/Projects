import logging

from fastapi import FastAPI

from core.logger_config import setup_logging
from routers import collection, document
from src.core.weaviate_client import WeaviateClient

setup_logging()
logger = logging.getLogger("app")

app = FastAPI(title="Practice Project", description="Created for practicing reasons.")
logger.info("Application starting")

weaviate_client = WeaviateClient()
app.state.client = weaviate_client

app.include_router(collection.router)
app.include_router(document.router)