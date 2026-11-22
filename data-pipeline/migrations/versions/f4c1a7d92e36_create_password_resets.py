"""create password_resets: one-time "Forgot password?" links

Revision ID: f4c1a7d92e36
Revises: e7b3c9a41f02
Create Date: 2026-11-22 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f4c1a7d92e36"
down_revision: str | Sequence[str] | None = "e7b3c9a41f02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "password_resets",
        sa.Column("reset_id", sa.BigInteger(), nullable=False),
        sa.Column("token_hash", sa.LargeBinary(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("reset_id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("password_resets_user_idx", "password_resets", ["user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("password_resets")
