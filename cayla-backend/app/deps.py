"""The login check + role check that every API uses.

Usage in any router (yours or Yee Mon's):

    from app.deps import CurrentUser, AdminUser

    @router.get("/something")
    def anyone_logged_in(user: CurrentUser): ...

    @router.post("/something")
    def admins_only(user: AdminUser): ...

For "TBC" endpoints (rerun-vlm, export), start with CurrentUser and swap to
AdminUser once the team decides; it is a one-word change.

Someone still on a temporary password is logged in but blocked (403
"Password change required") from everything that uses CurrentUser or
AdminUser. Only /me, logout and change-password use AnyLoggedInUser, so the
dashboard can send them to the "choose your password" screen.
"""

from typing import Annotated

from fastapi import Cookie, Depends, Header, HTTPException, status
from psycopg import Connection
from pydantic import BaseModel

from .config import SESSION_COOKIE
from .db import get_conn
from .security import token_digest


class User(BaseModel):
    user_id: int
    name: str
    email: str
    role: str  # 'admin' | 'staff'
    staff_id: str | None = None
    must_change_password: bool = False  # true = still on the emailed temporary password

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


Conn = Annotated[Connection, Depends(get_conn)]


def extract_token(cookie: str | None, authorization: str | None) -> str | None:
    # Dashboard uses the httpOnly cookie; scripts/tests can send "Authorization: Bearer <token>".
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return cookie


def get_session_user(
    conn: Conn,
    cctv_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """Anyone with a valid session, including someone still on a temporary password."""
    token = extract_token(cctv_session, authorization)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")

    row = conn.execute(
        """
        SELECT u.user_id, u.name, u.email, u.role, u.staff_id, u.must_change_password
        FROM sessions s
        JOIN users u ON u.user_id = s.user_id
        WHERE s.token_hash = %s
          AND s.revoked_at IS NULL
          AND s.expires_at > now()
          AND u.is_active
        """,
        (token_digest(token),),
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired or invalid")
    return User(**row)


AnyLoggedInUser = Annotated[User, Depends(get_session_user)]


def get_current_user(user: AnyLoggedInUser) -> User:
    if user.must_change_password:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Password change required")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
