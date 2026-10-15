"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Self

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = Field(default="gemma3:4b", min_length=1)
    ollama_base_url: HttpUrl = HttpUrl("http://localhost:11434")
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Context window requested from Ollama, and the cap on each response. Both are in tokens.
    ollama_num_ctx: int = Field(default=16_384, ge=2_048, le=131_072)
    ollama_num_predict: int = Field(default=2_048, ge=256, le=8_192)
    researchpilot_max_analysis_passes: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def check_response_cap_fits_context(self) -> Self:
        # The report prompt carries earlier model output (notes and gaps) plus room for its
        # own response, so three response-sized slices must fit with space left for sources.
        if self.ollama_num_predict * 4 > self.ollama_num_ctx:
            raise ValueError("OLLAMA_NUM_PREDICT must be at most a quarter of OLLAMA_NUM_CTX.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for the current process."""
    return Settings()
