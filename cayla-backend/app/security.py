"""Password hashing and session tokens."""

import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_ph = PasswordHasher()

# Used when the email doesn't exist, so a wrong email takes as long as a wrong
# password (otherwise response time reveals which emails are registered).
_DUMMY_HASH = _ph.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return (
            _ph.verify(password_hash or _DUMMY_HASH, password)
            and password_hash is not None
        )
    except (VerificationError, InvalidHashError):
        return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> bytes:
    """What we store in sessions.token_hash. A DB leak doesn't leak live tokens."""
    return hashlib.sha256(token.encode()).digest()
