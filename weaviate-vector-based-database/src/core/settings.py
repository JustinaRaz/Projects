from functools import lru_cache
from pathlib import Path

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
        docs_path = self.db_dir / "docs"
        docs_path.mkdir(parents=True, exist_ok=True)
        return docs_path

    @property
    def logging_config_path(self) -> Path:
        return self.config_dir / "logger.yaml"

    @property
    def log_dir(self) -> Path:
        log_path = self.db_dir / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str


@lru_cache
def get_settings() -> Settings:
    return Settings()