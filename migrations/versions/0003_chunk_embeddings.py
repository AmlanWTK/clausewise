"""Add embedding column + HNSW index to chunks.

Revision ID: 0003
Revises: 0002

``embedding_model`` is stored per row: re-embedding with a different model is
a tracked migration, never a silent mystery. NULL embedding = not yet embedded
(the embed script is resumable on exactly this predicate).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIMENSIONS = 384  # BGE-small; must match Settings.embedding_dimensions


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding", sa.Text(), nullable=True))
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({DIMENSIONS}) USING NULL")
    op.add_column("chunks", sa.Column("embedding_model", sa.Text(), nullable=True))
    # HNSW: best recall/speed trade-off for our scale; cosine ops to match
    # normalized embeddings. NULL rows are simply absent from the index.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
