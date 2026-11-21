from sqlalchemy import case, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, joinedload

from data_pipeline.database.models import (
    Incident,
    Review,
    ReviewOutcome,
    Severity,
    VLMResult,
)
from data_pipeline.database.repositories.incidents import incident_filters

SEVERITY_RANK = case(
    (Incident.severity == Severity.CRITICAL, 0),
    (Incident.severity == Severity.MODERATE, 1),
    else_=2,
)


def list_review_queue(
    db: Session,
    *,
    location_id: int | None = None,
    camera_id: str | None = None,
    event_type: str | None = None,
    severity: Severity | None = None,
    limit: int,
    offset: int = 0,
) -> tuple[list[tuple[Incident, VLMResult | None]], int]:
    """Read one page of incidents waiting for review, with the total count.

    Args:
        db: Session whose transaction is managed by the caller.
        location_id: Keep only incidents at this location.
        camera_id: Keep only incidents from this camera.
        event_type: Keep only incidents of this event type.
        severity: Keep only incidents with this severity.
        limit: Maximum number of incidents to return.
        offset: Number of matching incidents to skip.

    Returns:
        A (rows, total) tuple. Each row is (incident, newest VLM result or
        None). Critical incidents come first, then moderate, then minor;
        within a severity the longest-waiting incident comes first.
    """
    latest_vlm_id = (
        select(VLMResult.id)
        .where(VLMResult.incident_id == Incident.id)
        .order_by(VLMResult.processed_at.desc(), VLMResult.id.desc())
        .limit(1)
        .correlate(Incident)
        .scalar_subquery()
    )
    conditions = [
        Review.id.is_(None),
        *incident_filters(
            location_id=location_id,
            camera_id=camera_id,
            event_type=event_type,
            severity=severity,
        ),
    ]
    total = db.scalar(
        select(func.count(Incident.id))
        .outerjoin(Review, Review.incident_id == Incident.id)
        .where(*conditions)
    )
    statement = (
        select(Incident, VLMResult)
        .outerjoin(Review, Review.incident_id == Incident.id)
        .outerjoin(VLMResult, VLMResult.id == latest_vlm_id)
        .where(*conditions)
        .order_by(SEVERITY_RANK, Incident.first_anomaly_at, Incident.id)
        .limit(limit)
        .offset(offset)
    )
    rows = [(incident, vlm) for incident, vlm in db.execute(statement)]
    return rows, total


def create_review(
    db: Session,
    *,
    incident_id: int,
    outcome: ReviewOutcome,
    notes: str | None,
    corrected_type: str | None,
    reviewer_id: int,
) -> Review | None:
    """Insert an incident's review without committing.

    Args:
        db: Session whose transaction is managed by the caller.
        incident_id: Existing incident primary key.
        outcome: Reviewer's decision.
        notes: Optional reviewer notes.
        corrected_type: Event type the reviewer says it really was, or None.
        reviewer_id: User submitting the review.

    Returns:
        The new review, or None if the incident already has one.
    """
    statement = (
        insert(Review)
        .values(
            incident_id=incident_id,
            outcome=outcome,
            notes=notes,
            corrected_type=corrected_type,
            reviewer_id=reviewer_id,
        )
        .on_conflict_do_nothing(index_elements=[Review.incident_id])
        .returning(Review)
    )
    return db.scalars(statement).one_or_none()


def get_review(db: Session, review_id: int) -> Review | None:
    """Read a review with its incident and settlement loaded, or None."""
    statement = (
        select(Review)
        .where(Review.id == review_id)
        .options(joinedload(Review.incident).joinedload(Incident.settlement))
    )
    return db.scalars(statement).one_or_none()


def update_review(
    db: Session, review: Review, *, updated_by: int, **changes: object
) -> Review:
    """Apply changes to a review and stamp who edited it, without committing.

    Args:
        db: Session whose transaction is managed by the caller.
        review: Review to change.
        updated_by: User making the edit.
        **changes: New values for outcome, notes and corrected_type.

    Returns:
        The review, refreshed so updated_at holds the database time.
    """
    for name, value in changes.items():
        setattr(review, name, value)
    review.updated_at = func.now()
    review.updated_by = updated_by
    db.flush()
    db.refresh(review)
    return review
