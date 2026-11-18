"""#4 GET /users · #5 POST /users · #6 PATCH /users/{id} · POST /users/{id}/reset-password

This file is the worked example of the edit + audit pattern. Locations, cameras,
rules, settings and event types all follow the same shape.

Adding staff: the admin never types or sees a password. POST /users makes a
random temporary password, emails it to the new staff member, and marks the
account must_change_password until they choose their own at first login.
"""

from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, EmailStr, Field

from .. import audit
from ..config import APP_URL, TEMP_PASSWORD_HOURS
from ..deps import AdminUser, Conn
from ..mailer import EmailError, invite_email, reset_email, send_email
from ..password_policy import new_temp_password
from ..security import hash_password

router = APIRouter(prefix="/users", tags=["Login & users"])

Role = Literal["admin", "staff"]
USER_COLS = (
    "user_id, name, email, staff_id, date_of_birth, role, is_active, "
    "must_change_password, temp_password_expires_at, created_at, updated_at"
)


class UserOut(BaseModel):
    user_id: int
    name: str
    email: str
    staff_id: str | None
    date_of_birth: date | None
    role: Role
    is_active: bool
    must_change_password: bool  # true = invite pending (hasn't set a password yet)
    temp_password_expires_at: datetime | None  # when the emailed password stops working


class UserCreate(BaseModel):
    name: str = Field(min_length=1)
    email: EmailStr
    staff_id: str | None = Field(default=None, min_length=1)  # optional
    date_of_birth: date | None = None  # optional
    role: Role = "staff"  # staff by default; no password field: one is emailed


def _send_or_fail(email, to: str) -> None:
    """Send, or raise 502 so the whole request (user row + audit entry) is rolled back."""
    try:
        send_email(to, email)
    except EmailError:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Couldn't send the email, so nothing was saved. Check the mail settings and try again.",
        )


class UserUpdate(BaseModel):
    # Every field optional: PATCH only changes what is sent.
    name: str | None = Field(default=None, min_length=1)
    email: EmailStr | None = None
    staff_id: str | None = Field(default=None, min_length=1)
    date_of_birth: date | None = None
    role: Role | None = None
    is_active: bool | None = (
        None  # false = deactivate (users are never deleted: audit history points at them)
    )


@router.get("", response_model=list[UserOut])
def list_users(user: AdminUser, conn: Conn):
    return conn.execute(
        f"SELECT {USER_COLS} FROM users ORDER BY is_active DESC, name"
    ).fetchall()


@router.post("", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, user: AdminUser, conn: Conn):
    temp_password = new_temp_password()
    try:
        new = conn.execute(
            f"""
            INSERT INTO users (name, email, staff_id, date_of_birth, role, password_hash,
                               must_change_password, temp_password_expires_at, updated_by)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE, now() + %s * interval '1 hour', %s)
            RETURNING {USER_COLS}
            """,
            (
                body.name,
                body.email,
                body.staff_id,
                body.date_of_birth,
                body.role,
                hash_password(temp_password),
                TEMP_PASSWORD_HOURS,
                user.user_id,
            ),
        ).fetchone()
    except UniqueViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Email or staff ID already in use"
        )

    audit.record(
        conn,
        actor_id=user.user_id,
        action="create",
        entity_type="user",
        entity_id=new["user_id"],
        new=new,
    )
    # Last step: if the email fails, the user and audit entry above are rolled back.
    _send_or_fail(
        invite_email(
            new["name"],
            new["email"],
            temp_password,
            user.name,
            new["role"],
            TEMP_PASSWORD_HOURS,
            APP_URL,
        ),
        new["email"],
    )
    return new


@router.post("/{user_id}/reset-password", response_model=UserOut)
def reset_password(user_id: int, user: AdminUser, conn: Conn):
    """Email a new temporary password. Two uses, one API:
      - "Resend invite" (they never signed in, or the invite expired): sends the invite email again
      - "Reset password" (forgot it): sends the reset email
    Logs the person out everywhere; they must choose a new password at next login."""
    if user_id == user.user_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Use change-password for your own account"
        )
    old = conn.execute(
        f"SELECT {USER_COLS} FROM users WHERE user_id = %s FOR UPDATE", (user_id,)
    ).fetchone()
    if old is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if not old["is_active"]:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Reactivate this account first"
        )

    temp_password = new_temp_password()
    new = conn.execute(
        f"""
        UPDATE users SET password_hash = %s, must_change_password = TRUE,
               temp_password_expires_at = now() + %s * interval '1 hour',
               updated_at = now(), updated_by = %s
        WHERE user_id = %s RETURNING {USER_COLS}
        """,
        (hash_password(temp_password), TEMP_PASSWORD_HOURS, user.user_id, user_id),
    ).fetchone()
    conn.execute(
        "UPDATE sessions SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL",
        (user_id,),
    )
    audit.record(
        conn,
        actor_id=user.user_id,
        action="reset_password",
        entity_type="user",
        entity_id=user_id,
        old=old,
        new=new,
    )
    if old["must_change_password"]:  # never set a password yet: this is a resent invite
        email = invite_email(
            new["name"],
            new["email"],
            temp_password,
            user.name,
            new["role"],
            TEMP_PASSWORD_HOURS,
            APP_URL,
        )
    else:
        email = reset_email(
            new["name"],
            new["email"],
            temp_password,
            user.name,
            TEMP_PASSWORD_HOURS,
            APP_URL,
        )
    _send_or_fail(email, new["email"])
    return new


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: int, body: UserUpdate, user: AdminUser, conn: Conn):
    changes = body.model_dump(exclude_unset=True)

    # Guard against an admin locking everyone out by demoting/deactivating themselves.
    if user_id == user.user_id and (
        changes.get("role") == "staff" or changes.get("is_active") is False
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "You can't demote or deactivate your own account",
        )

    # 1. Read the current row (FOR UPDATE locks it until this request's transaction ends).
    old = conn.execute(
        f"SELECT {USER_COLS} FROM users WHERE user_id = %s FOR UPDATE", (user_id,)
    ).fetchone()
    if old is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if not changes:
        return old

    # 2. Apply the change. Column names come from the Pydantic model, never from the client.
    set_sql = ", ".join(f"{col} = %s" for col in changes)
    try:
        new = conn.execute(
            f"UPDATE users SET {set_sql}, updated_at = now(), updated_by = %s "
            f"WHERE user_id = %s RETURNING {USER_COLS}",
            (*changes.values(), user.user_id, user_id),
        ).fetchone()
    except UniqueViolation:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Email or staff ID already in use"
        )

    # 3. Side effect: deactivating someone logs them out everywhere immediately.
    if changes.get("is_active") is False:
        conn.execute(
            "UPDATE sessions SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL",
            (user_id,),
        )

    # 4. Audit (same transaction as the update).
    audit.record(
        conn,
        actor_id=user.user_id,
        action="update",
        entity_type="user",
        entity_id=user_id,
        old=old,
        new=new,
    )
    return new
