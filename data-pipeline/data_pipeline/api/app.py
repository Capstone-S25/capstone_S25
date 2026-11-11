"""Build the FastAPI app.

Run from data-pipeline/ with ``npm run api``.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from data_pipeline.api.routers import incidents
from data_pipeline.database.session import create_engine_and_session_factory


def create_app(database_url: str | None = None) -> FastAPI:
    """Create the API app with its own database engine.

    Args:
        database_url: SQLAlchemy connection URL. Defaults to DATABASE_URL
            from the environment or data-pipeline/.env.

    Returns:
        The app. Its engine is disposed when the app shuts down.
    """
    if database_url is None:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "Set DATABASE_URL in data-pipeline/.env or your environment"
            )
    engine, session_factory = create_engine_and_session_factory(database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()

    app = FastAPI(title="CCTV Incident Platform API", lifespan=lifespan)
    app.state.session_factory = session_factory
    app.include_router(incidents.router)
    return app
