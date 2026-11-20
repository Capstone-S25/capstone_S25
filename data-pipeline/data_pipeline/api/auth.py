"""The signed-in user, for every incidents and reviews route.

Every route imports get_current_user from here. The session check is
cayla-backend's (app/deps.py): the cctv_session cookie that login sets, or
"Authorization: Bearer <token>", must belong to an unexpired, unrevoked
session of an active user. No session answers 401; a user still on an emailed
temporary password answers 403 "Password change required".
"""

from dataclasses import dataclass
from enum import StrEnum

from data_pipeline.api.accounts import SignedInUser


class Role(StrEnum):
    ADMIN = "admin"
    STAFF = "staff"


@dataclass(frozen=True)
class CurrentUser:
    """The logged-in user.

    Attributes:
        id: users.user_id, stored as reviewer_id, settled_by and updated_by.
        name: Display name.
        role: Role.ADMIN or Role.STAFF.
    """

    id: int
    name: str
    role: Role


def get_current_user(user: SignedInUser) -> CurrentUser:
    """Return the user of this request's session, or answer 401 or 403."""
    return CurrentUser(id=user.user_id, name=user.name, role=Role(user.role))
