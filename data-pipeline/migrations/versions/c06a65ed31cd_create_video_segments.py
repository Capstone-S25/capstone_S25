"""create video segments

Revision ID: c06a65ed31cd
Revises:
Create Date: 2026-10-12 23:17:51.045797

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c06a65ed31cd"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "video_segments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("start_at_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_at_ms", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "end_at_ms >= start_at_ms", name="ck_video_segments_time_order"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path", name="uq_video_segments_path"),
    )
    op.create_index(
        "ix_video_segments_source_start",
        "video_segments",
        ["source_id", "start_at_ms"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_video_segments_source_start", table_name="video_segments")
    op.drop_table("video_segments")
