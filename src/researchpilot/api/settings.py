"""Settings for the HTTP API, separate from the research engine's settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Validated API settings, read from ``RESEARCHPILOT_API_*`` environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="RESEARCHPILOT_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local-only by default; binding to another interface exposes the API to the network.
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65_535)
    # How long the readiness check waits for Ollama before reporting it unavailable.
    ready_timeout_seconds: float = Field(default=2.0, gt=0.0, le=10.0)
    # Largest accepted request body; larger ones get 413. The largest valid research request
    # (80,000 characters of source material) is about 330 KB of UTF-8.
    max_request_bytes: int = Field(default=1_048_576, ge=1_024, le=16_777_216)
    # Research jobs waiting behind the one running job; further submissions are refused.
    max_queued_jobs: int = Field(default=4, ge=1, le=100)
    # Finished jobs kept in memory; the oldest are forgotten first.
    max_retained_jobs: int = Field(default=50, ge=1, le=1_000)
    # How long shutdown waits for a running job before leaving it behind.
    shutdown_timeout_seconds: float = Field(default=5.0, ge=0.0, le=60.0)
