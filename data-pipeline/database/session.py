from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def create_engine_and_session_factory(
    database_url: str,
) -> tuple[Engine, sessionmaker[Session]]:
    """Create a database engine and a factory for short-lived sessions.

    Args:
        database_url: SQLAlchemy connection URL from application configuration.

    Returns:
        An (engine, factory) tuple. Use ``with factory.begin() as db`` to
        commit on success, roll back on error, and automatically close
        the session.

    Notes:
        Create once per application process and reuse across operations.
        Database connections are opened when needed, and pool_pre_ping
        checks pooled connections before reuse.

        Call ``engine.dispose()`` at shutdown, after workers have stopped
        and all sessions have closed.
    """
    engine = create_engine(database_url, pool_pre_ping=True)
    factory = sessionmaker(bind=engine)
    return engine, factory
