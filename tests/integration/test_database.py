"""Integration tests requiring a running Postgres (docker compose up + migrations).

Run with:  uv run pytest -m integration
Skipped automatically when the database is unreachable.
"""

import pytest
import sqlalchemy as sa

from clausewise.config import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine() -> sa.Engine:
    eng = sa.create_engine(get_settings().sync_database_url, poolclass=sa.NullPool)
    try:
        with eng.connect():
            pass
    except sa.exc.OperationalError:
        pytest.skip("Postgres is not running (docker compose -f docker/compose.yml up -d)")
    return eng


def test_can_connect(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT 1")).scalar_one() == 1


def test_pgvector_extension_installed(engine: sa.Engine) -> None:
    """Fails if migration 0001 has not been applied (alembic upgrade head)."""
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).one_or_none()
    assert row is not None, "pgvector extension missing — run: uv run alembic upgrade head"


def test_alembic_is_at_head(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        version = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
    assert version == "0001"
