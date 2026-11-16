"""Temporary passwords and the rules for a new password. Standard library only.

Rules for a new password: at least 8 characters, with a lower-case letter, a
capital letter, a number and a symbol (plus: not the same as the temporary password).
"""

import secrets

# No look-alike characters (0/O, 1/l/I), so a temporary password is easy to type from an email.
_LOWER = "abcdefghjkmnpqrstuvwxyz"
_UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_DIGITS = "23456789"
_ALPHABET = _LOWER + _UPPER + _DIGITS

MIN_LENGTH = 8


def new_temp_password() -> str:
    """Random temporary password shaped like the invite email: K7m-9qXp-2Lw (3-4-3).
    The dashes are its symbol; the loop makes sure it also has a lower-case letter,
    a capital and a number, so it always passes password_problem()."""
    while True:
        raw = "".join(secrets.choice(_ALPHABET) for _ in range(10))
        if (
            any(c in _LOWER for c in raw)
            and any(c in _UPPER for c in raw)
            and any(c in _DIGITS for c in raw)
        ):
            return f"{raw[:3]}-{raw[3:7]}-{raw[7:]}"


def password_problem(
    new_password: str, current_password: str | None = None
) -> str | None:
    """Return why a new password is not allowed, or None if it is fine.
    Checked in this order, so the person sees the first thing to fix."""
    if len(new_password) < MIN_LENGTH:
        return f"Password must be at least {MIN_LENGTH} characters"
    if not any(c.islower() for c in new_password):
        return "Password must include a lower-case letter"
    if not any(c.isupper() for c in new_password):
        return "Password must include a capital letter"
    if not any(c.isdigit() for c in new_password):
        return "Password must include a number"
    if all(c.isalnum() for c in new_password):
        return "Password must include a symbol, such as ! @ # or -"
    if current_password is not None and new_password == current_password:
        return "New password must be different from the temporary one"
    return None
