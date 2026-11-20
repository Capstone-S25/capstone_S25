"""Login, staff accounts and the audit log, from cayla-backend/.

cayla-backend's own routers (/api/v1/auth, /me, /users, /audit-log) are
mounted on this app, so one server and one database serve the dashboard, the
incidents and the accounts. The account tables are in
data_pipeline.database.models.accounts.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends
from psycopg import Cursor
from psycopg.rows import dict_row
from sqlalchemy.orm import Session

from data_pipeline.api.database import get_db


def _load_accounts_package() -> None:
    """Import cayla-backend/app as the package app, as cayla-backend does.

    It is a folder of this repository, not an installed package. Only app is
    loaded: putting the folder on sys.path would also expose its scripts
    package in place of data-pipeline's.
    """
    package = Path(__file__).resolve().parents[3] / "cayla-backend/app"
    loaded = sys.modules.get("app")
    if loaded is not None:
        if Path(loaded.__file__ or "").resolve().parent != package:
            raise ImportError(f"Another module named app is loaded: {loaded!r}")
        return
    spec = importlib.util.spec_from_file_location(
        "app", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["app"] = module
    spec.loader.exec_module(module)


_load_accounts_package()

from app import deps  # noqa: E402
from app.db import get_conn  # noqa: E402
from app.routers import audit_log, auth, users  # noqa: E402

PREFIX = "/api/v1"
ROUTERS = (auth.router, users.router, audit_log.router)

# A signed-in user who has chosen their own password; 401 or 403 otherwise.
SignedInUser = deps.CurrentUser


class RequestConnection:
    """cayla-backend's database connection, inside this request's transaction.

    Its SQL calls conn.execute(...) and reads rows as dicts. Running it on
    the SQLAlchemy session's own psycopg connection means one connection per
    request, and account changes commit or roll back with everything else.
    """

    def __init__(self, db: Session):
        self._connection = db.connection().connection.driver_connection

    def execute(self, query: Any, params: Any = None) -> Cursor:
        return self._connection.cursor(row_factory=dict_row).execute(query, params)


def get_request_connection(
    db: Annotated[Session, Depends(get_db)],
) -> RequestConnection:
    """Replaces cayla-backend's app.db.get_conn (its own connection pool)."""
    return RequestConnection(db)


__all__ = ["PREFIX", "ROUTERS", "SignedInUser", "get_conn", "get_request_connection"]
