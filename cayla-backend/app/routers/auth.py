"""#1 POST /auth/login · #2 POST /auth/logout · #3 GET /me · POST /auth/change-password
POST /auth/forgot-password · POST /auth/reset-password (self-service "Forgot password?")

First login for a new staff member:
  1. login with the emailed temporary password -> 200, user.must_change_password = true
  2. the dashboard shows "choose your password" (everything else answers 403 until then)
  3. POST /auth/change-password {new_password, confirm_password}
     -> must_change_password = false, full access ("Save and continue")
  Later, from Settings: the same API, plus current_password.

Forgot password (no sign-in needed):
  1. POST /auth/forgot-password {email} -> 202, the same answer whether or not
     the email has an account; if it has, a one-time link to
     /reset-password?token=... is emailed (RESET_LINK_MINUTES, default 60)
  2. POST /auth/reset-password {token, new_password, confirm_password} -> 204;
     the link stops working, and every session of that user is signed out
  The current password keeps working until the link is used, so someone who
  types another person's email can't lock them out.
"""

from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, Cookie, Header, HTTPException, Response, status
from pydantic import BaseModel

from .. import audit
from ..config import (
    COOKIE_SECURE,
    PASSWORD_RESET_URL,
    RESET_LINK_MINUTES,
    SESSION_COOKIE,
    SESSION_HOURS,
)
from ..deps import AnyLoggedInUser, Conn, User, extract_token
from ..mailer import EmailError, reset_link_email, send_email
from ..password_policy import password_problem
from ..security import hash_password, new_session_token, token_digest, verify_password

router = APIRouter(tags=["Login & users"])


class LoginIn(BaseModel):
    email: str
    password: str


class LoginOut(BaseModel):
    user: User  # check user.must_change_password: true -> show "choose your password"
    token: str  # also set as an httpOnly cookie; the dashboard can ignore this field
    expires_at: datetime


class ChangePasswordIn(BaseModel):
    new_password: str
    confirm_password: str | None = (
        None  # the "Confirm new password" box; checked if sent
    )
    current_password: str | None = (
        None  # not needed on the first-login "Set your password" screen
    )
    # (they just signed in with the temporary one); needed
    # when changing it later from Settings


class ForgotPasswordIn(BaseModel):
    email: str


class ResetPasswordIn(BaseModel):
    token: str  # from the emailed link: /reset-password?token=...
    new_password: str
    confirm_password: str | None = None  # checked if sent


@router.post("/auth/login", response_model=LoginOut)
def login(body: LoginIn, conn: Conn, response: Response):
    row = conn.execute(
        "SELECT user_id, name, email, role, staff_id, password_hash, is_active, "
        "       must_change_password, temp_password_expires_at "
        "FROM users WHERE lower(email) = lower(%s)",
        (body.email.strip(),),
    ).fetchone()

    # Always run verify (even for unknown emails) and give one generic error,
    # so the response doesn't reveal which emails exist or are deactivated.
    ok = verify_password(body.password, row["password_hash"] if row else None)
    if not ok or not row["is_active"]:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")

    # Only said to someone who already typed the right temporary password.
    expires = row["temp_password_expires_at"]
    if (
        row["must_change_password"]
        and expires is not None
        and expires < datetime.now(timezone.utc)
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Your temporary password has expired. Ask an admin to send a new one.",
        )

    token = new_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
        (token_digest(token), row["user_id"], expires_at),
    )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_HOURS * 3600,
    )
    user = User(
        **{
            k: row[k]
            for k in (
                "user_id",
                "name",
                "email",
                "role",
                "staff_id",
                "must_change_password",
            )
        }
    )
    return LoginOut(user=user, token=token, expires_at=expires_at)


@router.post("/auth/change-password", response_model=User)
def change_password(
    body: ChangePasswordIn,
    user: AnyLoggedInUser,
    conn: Conn,
    cctv_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    authorization: str | None = Header(default=None),
):
    row = conn.execute(
        "SELECT password_hash, must_change_password FROM users WHERE user_id = %s FOR UPDATE",
        (user.user_id,),
    ).fetchone()
    if not row["must_change_password"]:
        if not body.current_password or not verify_password(
            body.current_password, row["password_hash"]
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Current password is incorrect"
            )
    if body.confirm_password is not None and body.confirm_password != body.new_password:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Passwords don't match"
        )
    problem = password_problem(body.new_password)
    if problem:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, problem)
    # Compared against the stored hash, so it works without knowing the temporary password.
    if verify_password(body.new_password, row["password_hash"]):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "New password must be different from the temporary one"
            if row["must_change_password"]
            else "New password must be different from the current one",
        )

    conn.execute(
        "UPDATE users SET password_hash = %s, must_change_password = FALSE, "
        "temp_password_expires_at = NULL, updated_at = now(), updated_by = %s WHERE user_id = %s",
        (hash_password(body.new_password), user.user_id, user.user_id),
    )
    # Log out every other session (other PCs, an old browser); this one stays logged in.
    token = extract_token(cctv_session, authorization)
    conn.execute(
        "UPDATE sessions SET revoked_at = now() "
        "WHERE user_id = %s AND revoked_at IS NULL AND token_hash <> %s",
        (user.user_id, token_digest(token)),
    )
    # The password itself is never logged; only that it changed.
    audit.record(
        conn,
        actor_id=user.user_id,
        action="change_password",
        entity_type="user",
        entity_id=user.user_id,
        old={"must_change_password": row["must_change_password"]},
        new={"must_change_password": False},
    )
    return user.model_copy(update={"must_change_password": False})


@router.post("/auth/logout", status_code=204)
def logout(
    user: AnyLoggedInUser,
    conn: Conn,
    response: Response,
    cctv_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    authorization: str | None = Header(default=None),
):
    # Revokes only the session used for this request (other PCs stay logged in).
    token = extract_token(cctv_session, authorization)
    conn.execute(
        "UPDATE sessions SET revoked_at = now() WHERE token_hash = %s AND revoked_at IS NULL",
        (token_digest(token),),
    )
    response.delete_cookie(SESSION_COOKIE)


@router.get("/me", response_model=User)
def me(user: AnyLoggedInUser):
    return user


FORGOT_PASSWORD_ANSWER = {
    "detail": "If that email has an account, a link to reset the password is on its way."
}


@router.post("/auth/forgot-password", status_code=status.HTTP_202_ACCEPTED)
def forgot_password(body: ForgotPasswordIn, conn: Conn):
    """Email a one-time link to choose a new password. No sign-in needed.

    Always gives the same answer, so it can't be used to find out which emails
    have an account. At most one email a minute per account.
    """
    row = conn.execute(
        "SELECT user_id, name, email FROM users "
        "WHERE lower(email) = lower(%s) AND is_active",
        (body.email.strip(),),
    ).fetchone()
    if row is None:
        return FORGOT_PASSWORD_ANSWER
    recent = conn.execute(
        "SELECT 1 FROM password_resets "
        "WHERE user_id = %s AND created_at > now() - interval '1 minute'",
        (row["user_id"],),
    ).fetchone()
    if recent is not None:
        return FORGOT_PASSWORD_ANSWER

    token = new_session_token()
    # Only the newest link works: earlier unused ones are closed.
    conn.execute(
        "UPDATE password_resets SET used_at = now() "
        "WHERE user_id = %s AND used_at IS NULL",
        (row["user_id"],),
    )
    conn.execute(
        "INSERT INTO password_resets (token_hash, user_id, expires_at) "
        "VALUES (%s, %s, now() + %s * interval '1 minute')",
        (token_digest(token), row["user_id"], RESET_LINK_MINUTES),
    )
    audit.record(
        conn,
        actor_id=None,  # nobody is signed in
        action="request_password_reset",
        entity_type="user",
        entity_id=row["user_id"],
    )
    reset_url = f"{PASSWORD_RESET_URL}?{urlencode({'token': token})}"
    try:
        send_email(
            row["email"], reset_link_email(row["name"], reset_url, RESET_LINK_MINUTES)
        )
    except EmailError as error:
        # Not told to the caller: that would reveal that the email has an account.
        print(
            f"Password reset email for user {row['user_id']} failed: {error}",
            flush=True,
        )
    return FORGOT_PASSWORD_ANSWER


@router.post("/auth/reset-password", status_code=status.HTTP_204_NO_CONTENT)
def reset_password(body: ResetPasswordIn, conn: Conn):
    """Choose a new password with the emailed link, then sign in with it.

    The link works once. Every existing session of the user is signed out.
    """
    row = conn.execute(
        "SELECT r.reset_id, u.user_id, u.must_change_password "
        "FROM password_resets r JOIN users u ON u.user_id = r.user_id "
        "WHERE r.token_hash = %s AND r.used_at IS NULL AND r.expires_at > now() "
        "  AND u.is_active "
        "FOR UPDATE OF r, u",
        (token_digest(body.token),),
    ).fetchone()
    if row is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This link has expired or has already been used. "
            "Ask for a new one from the sign-in page.",
        )
    if body.confirm_password is not None and body.confirm_password != body.new_password:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Passwords don't match"
        )
    problem = password_problem(body.new_password)
    if problem:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, problem)

    user_id = row["user_id"]
    conn.execute(
        "UPDATE users SET password_hash = %s, must_change_password = FALSE, "
        "temp_password_expires_at = NULL, updated_at = now(), updated_by = %s "
        "WHERE user_id = %s",
        (hash_password(body.new_password), user_id, user_id),
    )
    conn.execute(
        "UPDATE password_resets SET used_at = now() WHERE reset_id = %s",
        (row["reset_id"],),
    )
    conn.execute(
        "UPDATE sessions SET revoked_at = now() "
        "WHERE user_id = %s AND revoked_at IS NULL",
        (user_id,),
    )
    audit.record(
        conn,
        actor_id=user_id,
        action="reset_password_by_link",
        entity_type="user",
        entity_id=user_id,
        old={"must_change_password": row["must_change_password"]},
        new={"must_change_password": False},
    )
