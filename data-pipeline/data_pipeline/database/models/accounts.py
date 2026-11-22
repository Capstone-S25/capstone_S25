from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    Text,
    false,
    func,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from data_pipeline.database.base import Base

# The login, staff-account and audit-log routes come from cayla-backend/ and
# use these tables through their own SQL (cayla-backend/app). The columns match
# cayla-backend/db/migrations/002_platform.sql and 004_temp_passwords.sql, so
# the accounts share this database with the incidents.


class User(Base):
    """A dashboard account.

    Attributes:
        user_id: Auto-generated primary key, stored as reviewer_id,
            settled_by and updated_by on incidents.
        name: Display name.
        email: Sign-in email, unique ignoring case.
        role: "admin" or "staff".
        is_active: False once deactivated; users are never deleted.
        staff_id: Optional unique staff number.
        date_of_birth: Optional, for password-locked report PDFs.
        password_hash: argon2 hash, or None before one is set.
        created_at: Creation time, set by the database.
        updated_at: Time of the last edit.
        updated_by: User who made the last edit.
        must_change_password: True while on an emailed temporary password.
        temp_password_expires_at: When the temporary password stops working.
    """

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'staff')", name="users_role_check"),
        Index("users_email_uq", func.lower(text("email")), unique=True),
        Index(
            "users_staff_id_uq",
            "staff_id",
            unique=True,
            postgresql_where=text("staff_id IS NOT NULL"),
        ),
    )

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())
    staff_id: Mapped[str | None] = mapped_column(Text)
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    password_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_by: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.user_id")
    )
    must_change_password: Mapped[bool] = mapped_column(Boolean, server_default=false())
    temp_password_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class UserSession(Base):
    """One sign-in. Only the SHA-256 of the session token is stored.

    Attributes:
        session_id: Auto-generated primary key.
        token_hash: SHA-256 digest of the cookie's token.
        user_id: Signed-in user; deleted with them.
        created_at: Sign-in time, set by the database.
        expires_at: When the session stops working.
        revoked_at: Sign-out (or forced sign-out) time, or None.
    """

    __tablename__ = "sessions"
    __table_args__ = (Index("sessions_user_idx", "user_id"),)

    session_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, unique=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLogEntry(Base):
    """One change, with only the fields that changed.

    Attributes:
        audit_id: Auto-generated primary key.
        occurred_at: Time of the change, set by the database.
        actor_user_id: Who made it, or None for the system.
        action: For example "create", "update" or "change_password".
        entity_type: For example "user".
        entity_id: Primary key of the changed row, as text.
        old_values: Changed fields before, or None for a create.
        new_values: Changed fields after, or None for a delete.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("audit_entity_idx", "entity_type", "entity_id"),
        Index("audit_occurred_idx", text("occurred_at DESC")),
        Index("audit_actor_idx", "actor_user_id"),
    )

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.user_id")
    )
    action: Mapped[str] = mapped_column(Text)
    entity_type: Mapped[str] = mapped_column(Text)
    entity_id: Mapped[str] = mapped_column(Text)
    old_values: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    new_values: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class PasswordReset(Base):
    """One emailed "Forgot password?" link. Only the SHA-256 of its token is stored.

    Attributes:
        reset_id: Auto-generated primary key.
        token_hash: SHA-256 digest of the link's token.
        user_id: Whose password it resets; deleted with them.
        created_at: When it was requested, set by the database.
        expires_at: When the link stops working.
        used_at: When it was used, or replaced by a newer link; None while open.
    """

    __tablename__ = "password_resets"
    __table_args__ = (Index("password_resets_user_idx", "user_id"),)

    reset_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, unique=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
