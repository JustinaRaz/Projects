import logging

from fastapi import FastAPI

from routers import collection, document
from src.core.logger_config import setup_logging
from src.core.settings import get_settings
from src.core.weaviate_client import WeaviateClient

settings = get_settings()
settings.init_dirs()

setup_logging()
logger = logging.getLogger("app")

app = FastAPI(title="Practice Project", description="Created for practicing reasons.")
logger.info("Application starting")

weaviate_client = WeaviateClient()
app.state.client = weaviate_client

app.include_router(collection.router)
app.include_router(document.router)