"""create accounts: users, sessions and audit_log

The login and staff-account tables of cayla-backend/db/migrations (002, 004),
in the incidents database so both share one database.

Revision ID: e7b3c9a41f02
Revises: d41f7c2a9b10
Create Date: 2026-11-20 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e7b3c9a41f02"
down_revision: str | Sequence[str] | None = "d41f7c2a9b10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("staff_id", sa.Text(), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_by", sa.BigInteger(), nullable=True),
        sa.Column(
            "must_change_password",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "temp_password_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.CheckConstraint("role IN ('admin', 'staff')", name="users_role_check"),
        sa.ForeignKeyConstraint(["updated_by"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("users_email_uq", "users", [sa.text("lower(email)")], unique=True)
    op.create_index(
        "users_staff_id_uq",
        "users",
        ["staff_id"],
        unique=True,
        postgresql_where=sa.text("staff_id IS NOT NULL"),
    )
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.BigInteger(), nullable=False),
        sa.Column("token_hash", sa.LargeBinary(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("sessions_user_idx", "sessions", ["user_id"])
    op.create_table(
        "audit_log",
        sa.Column("audit_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("actor_user_id", sa.BigInteger(), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("old_values", postgresql.JSONB(), nullable=True),
        sa.Column("new_values", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("audit_id"),
    )
    op.create_index("audit_entity_idx", "audit_log", ["entity_type", "entity_id"])
    op.create_index("audit_occurred_idx", "audit_log", [sa.text("occurred_at DESC")])
    op.create_index("audit_actor_idx", "audit_log", ["actor_user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("audit_log")
    op.drop_table("sessions")
    op.drop_table("users")
