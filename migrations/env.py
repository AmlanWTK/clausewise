"""Alembic migration environment.

Migrations run on a synchronous engine (psycopg driver) even though the app
runs async (asyncpg): migrations are a short-lived release step, and the sync
path is simpler and battle-tested. The URL is derived from application
settings so there is exactly one source of configuration.
"""

from alembic import context
from sqlalchemy import create_engine, pool

from clausewise.config import get_settings

# Model metadata for autogenerate support. Populated from Checkpoint 7 onward
# when ORM table definitions exist; None is correct until then.
target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (--sql mode)."""
    context.configure(
        url=get_settings().sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the configured database."""
    engine = create_engine(get_settings().sync_database_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
