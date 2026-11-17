from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from data_pipeline.api.auth import CurrentUser, get_current_user
from data_pipeline.api.database import get_db
from data_pipeline.api.schemas.reviews import (
    Review,
    ReviewCreate,
    ReviewQueueItem,
    ReviewQueuePage,
    ReviewUpdate,
    VLMHint,
)
from data_pipeline.database.models import ReviewOutcome, Severity
from data_pipeline.database.repositories import incidents as incident_repository
from data_pipeline.database.repositories import reviews as repository

router = APIRouter(
    prefix="/api/v1",
    tags=["reviews"],
    dependencies=[Depends(get_current_user)],
)

DbSession = Annotated[Session, Depends(get_db)]
User = Annotated[CurrentUser, Depends(get_current_user)]


@router.get("/reviews/queue", response_model=ReviewQueuePage)
def list_review_queue(
    db: DbSession,
    location_id: int | None = None,
    camera_id: str | None = None,
    event_type: str | None = None,
    severity: Severity | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ReviewQueuePage:
    """List incidents waiting for review, most urgent first (API List #21).

    Critical incidents come first, then the longest-waiting ones. Each
    item carries the VLM's verdict as a hint.
    """
    rows, total = repository.list_review_queue(
        db,
        location_id=location_id,
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    items = [
        ReviewQueueItem(
            id=incident.id,
            camera_id=incident.camera_id,
            location_id=incident.location_id,
            event_type=incident.event_type,
            severity=incident.severity,
            first_anomaly_at=incident.first_anomaly_at,
            last_anomaly_at=incident.last_anomaly_at,
            vlm_hint=VLMHint.model_validate(vlm) if vlm else None,
        )
        for incident, vlm in rows
    ]
    return ReviewQueuePage(items=items, total=total, page=page, page_size=page_size)


@router.post(
    "/incidents/{incident_id}/review",
    response_model=Review,
    status_code=status.HTTP_201_CREATED,
)
def submit_review(
    db: DbSession, user: User, incident_id: int, body: ReviewCreate
) -> Review:
    """Submit the review for an incident (#22).

    Each incident has one review; change it with PATCH /reviews/{id}.
    TODO: Alert admins about escalated incidents (Notifications sheet).
    """
    if not incident_repository.incident_exists(db, incident_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    review = repository.create_review(
        db,
        incident_id=incident_id,
        outcome=body.outcome,
        notes=body.notes,
        corrected_type=body.corrected_type,
        reviewer_id=user.id,
    )
    if review is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Incident already reviewed; edit the existing review instead",
        )
    return Review.model_validate(review)


@router.patch("/reviews/{review_id}", response_model=Review)
def edit_review(
    db: DbSession, user: User, review_id: int, body: ReviewUpdate
) -> Review:
    """Edit a submitted review, recording who edited it and when (#23).

    TODO: Write an audit log entry (old -> new) once Person A's helper exists.
    """
    review = repository.get_review(db, review_id)
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")
    changes = body.model_dump(exclude_unset=True)
    if (
        review.incident.settlement is not None
        and changes.get("outcome", ReviewOutcome.ESCALATED) != ReviewOutcome.ESCALATED
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Incident is already settled, so it must stay escalated",
        )
    review = repository.update_review(db, review, updated_by=user.id, **changes)
    return Review.model_validate(review)
