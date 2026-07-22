"""Create contracts and chunks tables.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "contracts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("parties", JSONB(), nullable=False, server_default="[]"),
        sa.Column("source", sa.Text(), nullable=False, server_default="CUAD"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("corpus", sa.Text(), nullable=False),
        sa.Column(
            "contract_id",
            sa.Text(),
            sa.ForeignKey("contracts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("section_path", JSONB(), nullable=False, server_default="[]"),
        sa.Column("clause_types", JSONB(), nullable=False, server_default="[]"),
        sa.Column("duplicate_of", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_chunks_corpus", "chunks", ["corpus"])
    op.create_index("ix_chunks_contract", "chunks", ["contract_id"])
    op.create_index("ix_chunks_corpus_contract", "chunks", ["corpus", "contract_id"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("contracts")
