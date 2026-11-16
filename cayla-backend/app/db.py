"""Connection pool (psycopg3, as in the deck) and a FastAPI dependency.

Each request gets one connection inside one transaction: if the handler raises,
everything (including the audit entry) rolls back together. That is what makes
"every edit is audited" reliable.
"""

from collections.abc import Iterator

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL

pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"row_factory": dict_row},
)


def get_conn() -> Iterator[Connection]:
    with pool.connection() as conn:  # commits on success, rolls back on exception
        yield conn
