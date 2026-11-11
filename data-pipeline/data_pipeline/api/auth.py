"""Temporary login used until Person A's real login and role check exist.

TODO(Person A): Replace this module's get_current_user with the real login
and role check. Every route imports get_current_user from here and nowhere
else, so the swap is one line: import the real dependency in place of the
fake one below (keeping the name get_current_user).
"""

import os
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    MEMBER = "member"


@dataclass(frozen=True)
class CurrentUser:
    """The logged-in user.

    Attributes:
        id: User primary key, stored as reviewer_id, settled_by and updated_by.
        name: Display name.
        role: Role.ADMIN or Role.MEMBER.
    """

    id: int
    name: str
    role: Role


def get_current_user() -> CurrentUser:
    """Return a hard-coded user for development.

    TODO(Person A): Replace with the real login and role check.

    The role comes from DEV_USER_ROLE ("admin" or "member"), defaulting to
    admin, so member-only behaviour can be tried without a login.
    """
    role = os.environ.get("DEV_USER_ROLE", Role.ADMIN)
    if role not in Role:
        raise RuntimeError(f"DEV_USER_ROLE must be admin or member, not {role!r}")
    return CurrentUser(id=1, name="Dev User", role=Role(role))
