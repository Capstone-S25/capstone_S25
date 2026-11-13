"""Build the FastAPI app.

Run from data-pipeline/ with ``npm run api``.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from data_pipeline.api.routers import evidence, incidents
from data_pipeline.database.session import create_engine_and_session_factory


def _serve_dashboard(app: FastAPI, directory: Path) -> None:
    """Serve the built React dashboard from the API's own port.

    The dashboard routes on the client, so any path that is not a file (for
    example /staff/home) returns index.html. API routes are registered first
    and keep priority.
    """

    @app.get("/{path:path}", include_in_schema=False)
    def dashboard(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        target = (directory / path).resolve()
        if path and target.is_relative_to(directory) and target.is_file():
            return FileResponse(target)
        return FileResponse(directory / "index.html")


def create_app(
    database_url: str | None = None, storage_root: Path | None = None
) -> FastAPI:
    """Create the API app with its own database engine.

    Args:
        database_url: SQLAlchemy connection URL. Defaults to DATABASE_URL
            from the environment or data-pipeline/.env.
        storage_root: Recordings root that incident evidence paths are
            relative to. Defaults to RECORDINGS_DIR, then
            data-pipeline/recordings.

    Returns:
        The app. Its engine is disposed when the app shuts down.
    """
    if database_url is None:
        # Same order as main.py: the environment, the root .env, then the
        # backend .env, without overwriting values already set.
        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
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
    app.state.storage_root = Path(
        storage_root
        or os.environ.get("RECORDINGS_DIR")
        or Path(__file__).resolve().parents[2] / "recordings"
    )
    app.include_router(incidents.router)
    app.include_router(evidence.router)
    dashboard = os.environ.get("DASHBOARD_DIR")
    if dashboard and (Path(dashboard) / "index.html").is_file():
        _serve_dashboard(app, Path(dashboard).resolve())
    return app
