from datetime import datetime
from typing import Literal

from sqlalchemy import ColumnElement, exists, func, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session, joinedload, selectinload

from data_pipeline.database.models import (
    Detection,
    Incident,
    Review,
    ReviewOutcome,
    Severity,
    VLMResult,
)

UNREVIEWED = "unreviewed"


def incident_filters(
    *,
    location_id: int | None = None,
    camera_id: str | None = None,
    event_type: str | None = None,
    severity: Severity | None = None,
) -> list[ColumnElement[bool]]:
    """Build WHERE conditions on Incident for each filter that is not None."""
    conditions = []
    if location_id is not None:
        conditions.append(Incident.location_id == location_id)
    if camera_id is not None:
        conditions.append(Incident.camera_id == camera_id)
    if event_type is not None:
        conditions.append(Incident.event_type == event_type)
    if severity is not None:
        conditions.append(Incident.severity == severity)
    return conditions


def list_incidents(
    db: Session,
    *,
    location_id: int | None = None,
    camera_id: str | None = None,
    event_type: str | None = None,
    severity: Severity | None = None,
    outcome: ReviewOutcome | Literal["unreviewed"] | None = None,
    occurred_from: datetime | None = None,
    occurred_before: datetime | None = None,
    limit: int,
    offset: int = 0,
) -> tuple[list[tuple[Incident, ReviewOutcome | None]], int]:
    """Read one page of incidents, newest first, with the total match count.

    Args:
        db: Session whose transaction is managed by the caller.
        location_id: Keep only incidents at this location.
        camera_id: Keep only incidents from this camera.
        event_type: Keep only incidents of this event type.
        severity: Keep only incidents with this severity.
        outcome: Keep only incidents whose review has this outcome, or
            "unreviewed" for incidents without a review.
        occurred_from: Keep incidents first detected at or after this time.
        occurred_before: Keep incidents first detected before this time.
        limit: Maximum number of incidents to return.
        offset: Number of matching incidents to skip.

    Returns:
        A (rows, total) tuple. Each row is (incident, review outcome or None),
        ordered by first_anomaly_at then id, both descending. total counts
        every match, ignoring limit and offset.
    """
    conditions = incident_filters(
        location_id=location_id,
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
    )
    if outcome == UNREVIEWED:
        conditions.append(Review.id.is_(None))
    elif outcome is not None:
        conditions.append(Review.outcome == outcome)
    if occurred_from is not None:
        conditions.append(Incident.first_anomaly_at >= occurred_from)
    if occurred_before is not None:
        conditions.append(Incident.first_anomaly_at < occurred_before)

    total = db.scalar(
        select(func.count(Incident.id))
        .outerjoin(Review, Review.incident_id == Incident.id)
        .where(*conditions)
    )
    statement = (
        select(Incident, Review.outcome)
        .outerjoin(Review, Review.incident_id == Incident.id)
        .where(*conditions)
        .order_by(Incident.first_anomaly_at.desc(), Incident.id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = [(incident, outcome) for incident, outcome in db.execute(statement)]
    return rows, total


def get_incident(db: Session, incident_id: int) -> Incident | None:
    """Read one incident with its detections, VLM results and review loaded.

    Args:
        db: Session whose transaction is managed by the caller.
        incident_id: Incident primary key.

    Returns:
        The incident, or None if it does not exist. detections are ordered by
        detected_at and vlm_results by processed_at, oldest first.
    """
    statement = (
        select(Incident)
        .where(Incident.id == incident_id)
        .options(
            selectinload(Incident.detections),
            selectinload(Incident.vlm_results),
            joinedload(Incident.review),
        )
    )
    return db.scalars(statement).one_or_none()


def incident_exists(db: Session, incident_id: int) -> bool:
    """Return True if an incident with this primary key exists."""
    return db.scalar(select(exists().where(Incident.id == incident_id)))


def request_vlm_rerun(
    db: Session, incident_id: int, *, requested_by: int
) -> Incident | None:
    """Ask for the VLM to check an incident again, without committing.

    A request that is already waiting is kept unchanged, so asking twice
    does not move it back in the VLM worker's queue.

    Args:
        db: Session whose transaction is managed by the caller.
        incident_id: Incident primary key.
        requested_by: User making the request.

    Returns:
        The incident with vlm_rerun_requested_at set, or None if it does
        not exist.
    """
    incident = db.get(Incident, incident_id)
    if incident is None:
        return None
    if incident.vlm_rerun_requested_at is None:
        incident.vlm_rerun_requested_at = func.now()
        incident.vlm_rerun_requested_by = requested_by
        db.flush()
        db.refresh(incident)
    return incident


def latest_vlm_results():
    """Subquery of each incident's newest VLM result (DISTINCT ON incident)."""
    return (
        select(VLMResult)
        .ext(distinct_on(VLMResult.incident_id))
        .order_by(
            VLMResult.incident_id,
            VLMResult.processed_at.desc(),
            VLMResult.id.desc(),
        )
        .subquery()
    )


def find_similar_incidents(
    db: Session, incident_id: int, *, limit: int
) -> list[tuple[Incident, ReviewOutcome | None, float]]:
    """Find past incidents whose clips look most like this one's.

    Compares the embedding on each incident's newest VLM result using
    pgvector cosine distance. Incidents without an embedding are skipped.

    Args:
        db: Session whose transaction is managed by the caller.
        incident_id: Incident to compare against.
        limit: Maximum number of incidents to return.

    Returns:
        (incident, review outcome or None, similarity) tuples, most similar
        first, where similarity is 1 - cosine distance (1 means identical).
        Empty if the incident has no embedding or does not exist.
    """
    latest = latest_vlm_results()
    target = db.scalar(
        select(latest.c.embedding).where(
            latest.c.incident_id == incident_id, latest.c.embedding.is_not(None)
        )
    )
    if target is None:
        return []
    distance = latest.c.embedding.cosine_distance(target)
    statement = (
        select(Incident, Review.outcome, distance)
        .join(latest, latest.c.incident_id == Incident.id)
        .outerjoin(Review, Review.incident_id == Incident.id)
        .where(Incident.id != incident_id, latest.c.embedding.is_not(None))
        .order_by(distance, Incident.id)
        .limit(limit)
    )
    return [
        (incident, outcome, 1 - distance)
        for incident, outcome, distance in db.execute(statement)
    ]


def get_evidence_start(db: Session, incident_id: int) -> datetime | None:
    """Return when an incident's evidence clip starts, or None if not found."""
    return db.scalar(
        select(Incident.evidence_start_at).where(Incident.id == incident_id)
    )


def list_flagged_detections(db: Session, incident_id: int) -> list[Detection]:
    """Read an incident's flagged detections for drawing during playback.

    Args:
        db: Session whose transaction is managed by the caller.
        incident_id: Incident primary key.

    Returns:
        Flagged detections ordered by detected_at, then id. Empty if the
        incident has none or does not exist; use get_evidence_start to tell
        the two apart.
    """
    statement = (
        select(Detection)
        .where(Detection.incident_id == incident_id, Detection.is_flagged)
        .order_by(Detection.detected_at, Detection.id)
    )
    return list(db.scalars(statement))
