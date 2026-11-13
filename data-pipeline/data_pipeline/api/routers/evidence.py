"""Evidence files for an incident: the clip (API List #17) and a thumbnail."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from data_pipeline.api.auth import get_current_user
from data_pipeline.api.database import get_db
from data_pipeline.database.models import Incident

router = APIRouter(
    prefix="/api/v1/incidents",
    tags=["incidents"],
    dependencies=[Depends(get_current_user)],
)

DbSession = Annotated[Session, Depends(get_db)]


def _evidence(request: Request, db: Session, incident_id: int, suffix: str) -> Path:
    relative = db.scalar(
        select(Incident.evidence_clip_path).where(Incident.id == incident_id)
    )
    if relative is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    root = Path(request.app.state.storage_root).resolve()
    path = (root / relative).with_suffix(suffix).resolve()
    # Stored paths are relative to the recordings root; never serve outside it.
    if not relative or not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Evidence file not found")
    return path


@router.get("/{incident_id}/clip", response_class=FileResponse)
def get_clip(request: Request, db: DbSession, incident_id: int) -> FileResponse:
    """Stream the incident's evidence clip (#17)."""
    return FileResponse(
        _evidence(request, db, incident_id, ".mp4"), media_type="video/mp4"
    )


@router.get("/{incident_id}/thumbnail", response_class=FileResponse)
def get_thumbnail(request: Request, db: DbSession, incident_id: int) -> FileResponse:
    """Return a still from the incident's evidence, for lists and cards."""
    return FileResponse(
        _evidence(request, db, incident_id, ".jpg"), media_type="image/jpeg"
    )
