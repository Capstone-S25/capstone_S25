from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session


def get_db(request: Request) -> Iterator[Session]:
    """Yield a session for one request.

    The transaction commits when the request succeeds and rolls back if
    it raises.
    """
    with request.app.state.session_factory.begin() as db:
        yield db
