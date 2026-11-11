from datetime import UTC, date, datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from data_pipeline.api.auth import get_current_user
from data_pipeline.api.database import get_db
from data_pipeline.api.schemas.incidents import (
    CameraRef,
    Detection,
    IncidentDetail,
    IncidentListItem,
    IncidentPage,
    LocationRef,
    OutcomeFilter,
    Review,
    VLMVerdict,
)
from data_pipeline.database.models import Severity
from data_pipeline.database.repositories import incidents as repository

router = APIRouter(
    prefix="/api/v1/incidents",
    tags=["incidents"],
    dependencies=[Depends(get_current_user)],
)

DbSession = Annotated[Session, Depends(get_db)]


def _start_of_day(day: date) -> datetime:
    # TODO: Use the site's time zone once settings exist; days are UTC for now.
    return datetime.combine(day, time.min, tzinfo=UTC)


@router.get("", response_model=IncidentPage)
def list_incidents(
    db: DbSession,
    location_id: int | None = None,
    camera_id: str | None = None,
    event_type: str | None = None,
    severity: Severity | None = None,
    outcome: OutcomeFilter | None = None,
    date_from: Annotated[date | None, Query(description="First day, inclusive")] = None,
    date_to: Annotated[date | None, Query(description="Last day, inclusive")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> IncidentPage:
    """List incidents, newest first (API List #15)."""
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "date_from must be on or before date_to",
        )
    rows, total = repository.list_incidents(
        db,
        location_id=location_id,
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
        outcome=outcome,
        occurred_from=_start_of_day(date_from) if date_from else None,
        occurred_before=(
            _start_of_day(date_to + timedelta(days=1)) if date_to else None
        ),
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    items = [
        IncidentListItem(
            id=incident.id,
            camera_id=incident.camera_id,
            location_id=incident.location_id,
            event_type=incident.event_type,
            severity=incident.severity,
            first_anomaly_at=incident.first_anomaly_at,
            last_anomaly_at=incident.last_anomaly_at,
            review_outcome=review_outcome,
        )
        for incident, review_outcome in rows
    ]
    return IncidentPage(items=items, total=total, page=page, page_size=page_size)


@router.get("/{incident_id}", response_model=IncidentDetail)
def get_incident(db: DbSession, incident_id: int) -> IncidentDetail:
    """Return one incident with everything the incident page shows (#16)."""
    incident = repository.get_incident(db, incident_id)
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    latest_vlm = incident.vlm_results[-1] if incident.vlm_results else None
    return IncidentDetail(
        id=incident.id,
        camera=CameraRef(id=incident.camera_id),
        location=LocationRef(id=incident.location_id),
        event_type=incident.event_type,
        severity=incident.severity,
        first_anomaly_at=incident.first_anomaly_at,
        last_anomaly_at=incident.last_anomaly_at,
        evidence_start_at=incident.evidence_start_at,
        evidence_end_at=incident.evidence_end_at,
        created_at=incident.created_at,
        detections=[
            Detection.from_model(detection, incident.evidence_start_at)
            for detection in incident.detections
        ],
        vlm_verdict=VLMVerdict.model_validate(latest_vlm) if latest_vlm else None,
        review=Review.model_validate(incident.review) if incident.review else None,
    )


@router.get("/{incident_id}/detections", response_model=list[Detection])
def list_flagged_detections(db: DbSession, incident_id: int) -> list[Detection]:
    """Return flagged YOLO boxes in time order, for drawing on the clip (#18)."""
    clip_start = repository.get_evidence_start(db, incident_id)
    if clip_start is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return [
        Detection.from_model(detection, clip_start)
        for detection in repository.list_flagged_detections(db, incident_id)
    ]
