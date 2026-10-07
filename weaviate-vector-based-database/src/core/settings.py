from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables and .env.

    Provides typed access to required config (e.g. Weaviate credentials)
    and derived filesystem paths used throughout the app, all resolved
    relative to the project root rather than the current working directory.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
        case_sensitive=True,
    )

    project_root: Path = PROJECT_ROOT

    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"

    @property
    def db_dir(self) -> Path:
        return self.project_root / "db"

    @property
    def doc_upload_dir(self) -> Path:
        docs_path = self.db_dir / "chunks"
        docs_path.mkdir(parents=True, exist_ok=True)
        return docs_path

    @property
    def text_upload_dir(self) -> Path:
        text_path = self.db_dir / "text"
        text_path.mkdir(parents=True, exist_ok=True)
        return text_path

    @property
    def logging_config_path(self) -> Path:
        return self.config_dir / "logger.yaml"

    @property
    def log_dir(self) -> Path:
        log_path = self.db_dir / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    def init_dirs(self) -> None:
        """Create all directories required by the application."""

        directories = [
            self.config_dir,
            self.src_dir,
            self.db_dir,
            self.doc_upload_dir,
            self.log_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    weaviate_url: str = Field(validation_alias="WEAVIATE_URL")
    weaviate_api_key: str = Field(validation_alias="WEAVIATE_API_KEY")

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS_PER_CHUNK: int = 200 # embedding model maximum
    OVERLAP_TOKENS: int = 40
    sentence_language: str = "en"


@lru_cache
def get_settings() -> Settings:
    return Settings()
