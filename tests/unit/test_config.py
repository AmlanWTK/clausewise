"""Unit tests for application settings."""

import pytest

from clausewise.config import Settings


def test_defaults_are_development_safe() -> None:
    settings = Settings(_env_file=None)  # ignore any local .env
    assert settings.environment == "development"
    assert not settings.is_production
    assert settings.log_level == "INFO"
    assert settings.database_url.startswith("postgresql+asyncpg://")


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")
    settings = Settings(_env_file=None)
    assert settings.is_production
    assert settings.log_level == "WARNING"
    assert settings.database_url == "postgresql+asyncpg://u:p@db:5432/x"


def test_sync_database_url_swaps_driver() -> None:
    settings = Settings(_env_file=None)
    assert settings.sync_database_url.startswith("postgresql+psycopg://")


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")  # not in our Literal
    with pytest.raises(ValueError, match="environment"):
        Settings(_env_file=None)
