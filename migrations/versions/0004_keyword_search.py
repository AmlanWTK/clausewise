"""Full-text search: generated tsvector column + GIN index on chunks.

Revision ID: 0004
Revises: 0003

A GENERATED column cannot drift from the text it indexes — the database
maintains it on every write, so keyword search is always consistent with
chunk content by construction.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tsv")
