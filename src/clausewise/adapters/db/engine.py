"""Engine and session management.

The ingestion pipeline runs synchronously (CPU-bound batch work — async adds
nothing); the API layer (Checkpoint 17) will use the async engine. Both derive
from the same Settings.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from clausewise.config import Settings


def sync_engine(settings: Settings) -> Engine:
    return create_engine(settings.sync_database_url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
