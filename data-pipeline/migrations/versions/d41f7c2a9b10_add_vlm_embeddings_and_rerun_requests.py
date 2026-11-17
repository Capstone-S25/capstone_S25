"""add vlm embeddings and re-run requests

Revision ID: d41f7c2a9b10
Revises: ba92658850ab
Create Date: 2026-11-17 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "d41f7c2a9b10"
down_revision: str | Sequence[str] | None = "ba92658850ab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("vlm_results", sa.Column("embedding", Vector(512), nullable=True))
    op.add_column(
        "incidents",
        sa.Column("vlm_rerun_requested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "incidents", sa.Column("vlm_rerun_requested_by", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("incidents", "vlm_rerun_requested_by")
    op.drop_column("incidents", "vlm_rerun_requested_at")
    op.drop_column("vlm_results", "embedding")
    op.execute("DROP EXTENSION IF EXISTS vector")
