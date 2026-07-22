"""Integration test: full pipeline against real Postgres, fixture data.

Requires: docker compose up + alembic upgrade head. Self-skips otherwise.
"""

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from clausewise.adapters.db.engine import create_session_factory, sync_engine
from clausewise.config import get_settings
from clausewise.ingestion.pipeline import run_ingestion

pytestmark = pytest.mark.integration

CONTRACT = (
    "ARTICLE I - DEFINITIONS\n\n"
    '1.1 Terms. "Agreement" means this document and all exhibits attached hereto, '
    "which together constitute the entire understanding between the parties.\n\n"
    "ARTICLE II - GOVERNING LAW\n\n"
    "2.1 Law. This Agreement shall be governed by the laws of the State of Delaware "
    "without regard to its conflict of laws provisions in any dispute arising here.\n"
)


@pytest.fixture(scope="module")
def factory() -> sessionmaker[Session]:
    settings = get_settings()
    engine = sync_engine(settings)
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM chunks LIMIT 1"))
    except sa.exc.OperationalError:
        pytest.skip("Postgres is not running")
    except sa.exc.ProgrammingError:
        pytest.skip("Schema missing — run: uv run alembic upgrade head")
    return create_session_factory(engine)


@pytest.fixture()
def fixture_data(tmp_path: Path) -> Path:
    law_start = CONTRACT.index("the laws of the State of Delaware")
    (tmp_path / "contracts.jsonl").write_text(
        json.dumps({"title": "IT_MSA_1", "text": CONTRACT}) + "\n", encoding="utf-8"
    )
    (tmp_path / "annotations.jsonl").write_text(
        json.dumps(
            {
                "title": "IT_MSA_1",
                "clause_type": "Governing Law",
                "question": "gov law?",
                "spans": [{"text": "the laws of the State of Delaware", "char_start": law_start}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _cleanup(factory: sessionmaker[Session]) -> None:
    with factory() as session:
        session.execute(
            sa.text(
                "DELETE FROM chunks WHERE contract_id IN "
                "(SELECT id FROM contracts WHERE title = 'IT_MSA_1')"
            )
        )
        session.execute(sa.text("DELETE FROM contracts WHERE title = 'IT_MSA_1'"))
        session.commit()


def test_ingest_persists_and_is_idempotent(
    factory: sessionmaker[Session], fixture_data: Path
) -> None:
    _cleanup(factory)
    try:
        first = run_ingestion(factory, corpora=["clause_aware"], data_dir=fixture_data)
        assert first.written_contracts == 1
        assert first.written_chunks > 0

        second = run_ingestion(factory, corpora=["clause_aware"], data_dir=fixture_data)
        assert second.written_contracts == 0, "re-run must be a no-op"
        assert second.written_chunks == 0, "re-run must be a no-op"

        with factory() as session:
            rows = session.execute(
                sa.text(
                    "SELECT clause_types FROM chunks c JOIN contracts k ON c.contract_id = k.id "
                    "WHERE k.title = 'IT_MSA_1' AND c.corpus = 'clause_aware'"
                )
            ).all()
        assert rows
        labels = {label for (types,) in rows for label in types}
        assert "Governing Law" in labels, "enrichment must reach the database"
    finally:
        _cleanup(factory)


def test_dry_run_writes_nothing(factory: sessionmaker[Session], fixture_data: Path) -> None:
    _cleanup(factory)
    stats = run_ingestion(factory, corpora=["clause_aware"], data_dir=fixture_data, dry_run=True)
    assert stats.chunks > 0
    assert stats.written_chunks == 0
    with factory() as session:
        count = session.execute(
            sa.text("SELECT count(*) FROM contracts WHERE title = 'IT_MSA_1'")
        ).scalar_one()
    assert count == 0
