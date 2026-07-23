"""Application configuration.

Single source of truth for all runtime settings. Values come from environment
variables (or a local ``.env`` file), validated and typed by pydantic-settings.

Usage: inject ``Settings`` via ``get_settings()`` at composition points (CLI
entry points, FastAPI dependencies). Never read ``os.environ`` elsewhere.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]


class Settings(BaseSettings):
    """Typed application settings, loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    environment: Environment = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Database ---
    # Port 5433 matches docker/compose.yml (5433 on the host to avoid clashing
    # with locally installed PostgreSQL instances on 5432).
    database_url: str = "postgresql+asyncpg://clausewise:clausewise@localhost:5433/clausewise"

    # --- Embeddings ---
    # 384-dim BGE-small: strong retrieval quality per parameter, CPU-friendly,
    # and fits the Supabase free tier. Must match migration 0003's vector(384).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    @property
    def sync_database_url(self) -> str:
        """Sync-driver variant of the database URL (Alembic migrations only)."""
        return self.database_url.replace("+asyncpg", "+psycopg")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()
