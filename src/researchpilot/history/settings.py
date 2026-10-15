"""Settings for the research history database, separate from the engine and API settings."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Validated settings, read from ``RESEARCHPILOT_DATABASE_*`` environment variables or .env.

    The URL is a secret: it is never logged or included in error messages. Its scheme is
    checked when an engine is created, so a malformed URL is never echoed by validation.
    """

    model_config = SettingsConfigDict(
        env_prefix="RESEARCHPILOT_DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # For example postgresql+psycopg://user:password@localhost:5432/researchpilot
    url: SecretStr | None = None
    connect_timeout_seconds: int = Field(default=3, ge=2, le=30)
    statement_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)
    lock_timeout_ms: int = Field(default=2_000, ge=100, le=60_000)
    pool_size: int = Field(default=5, ge=1, le=20)
    max_overflow: int = Field(default=5, ge=0, le=20)
    pool_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
