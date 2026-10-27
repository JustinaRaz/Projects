from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.settings import get_settings


class ModelSettings(BaseModel):
    """Describes and validates model-related settings.
    Any key that is not described in this class will raise a validation error.
    """
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str # required field
    device_map: str = "auto" # optional field with a default argument if not provided
    torch_dtype: Literal[
        "auto", "float32", "float16", "bfloat16"
    ] = "auto"
    cache_dir: Path = Path("db/gemma_model")


class GenerationSettings(BaseModel):
    """Describes and validates text generation settings.
    """
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=512, gt=0) # gt -> greater than 0.
    do_sample: bool = True # True - uses temperature and top_p
    temperature: float = Field(default=0.2, ge=0) # ge -> greater or equal to
    top_p: float = Field(default=0.9, gt=0, le=1)

    @model_validator(mode="after")
    def validate_sampling(self):
        if self.do_sample and self.temperature <= 0:
            error_message = "Temperature must be greater than 0 when sampling."
            raise ValueError(error_message)
        return self


class ContextSettings(BaseModel):
    """Describes and validates the LLM's context window.
    """
    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int = Field(default=5000, gt=0)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings
    generation: GenerationSettings
    context: ContextSettings


def load_llm_config() -> LLMConfig:
    """Load, validate, and resolve the LLM configuration."""

    settings = get_settings()

    with settings.llm_config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    config = LLMConfig.model_validate(raw_config)

    return config
