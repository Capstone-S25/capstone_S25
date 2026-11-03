from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from data_pipeline.database.models.video_segments import SegmentStatus, VideoSegment


def insert_video_segment_metadata(
    db: Session,
    *,
    source_id: str,
    session_id: str,
    storage_root: Path,
    path: Path,
    start_at_ms: int,
    end_at_ms: int,
    delete_at: datetime | None = None,
) -> bool:
    """Insert segment metadata in PostgreSQL without committing.

    Args:
        db: Session whose transaction is managed by the caller.
        source_id: Video source identifier.
        session_id: Recording session UUID as a hex or hyphenated string.
        storage_root: Base directory for recorded files.
        path: Segment file path; resolved relative to storage_root for storage.
            Relative input paths are resolved against the working directory.
        start_at_ms: Segment start as UTC epoch milliseconds.
        end_at_ms: Segment end as UTC epoch milliseconds.
        delete_at: Optional timezone-aware retention deadline. Defaults to None;

    Returns:
        True if inserted, or False if the normalized path already exists.
    """
    if delete_at is not None and delete_at.utcoffset() is None:
        raise ValueError("delete_at must be timezone-aware")

    relative_path = path.resolve().relative_to(storage_root.resolve()).as_posix()

    statement = (
        insert(VideoSegment)
        .values(
            source_id=source_id,
            session_id=UUID(session_id),
            path=relative_path,
            start_at_ms=start_at_ms,
            end_at_ms=end_at_ms,
            delete_at=delete_at,
        )
        .on_conflict_do_nothing(constraint="uq_video_segments_path")
        .returning(VideoSegment.id)
    )

    return db.execute(statement).scalar_one_or_none() is not None


def get_oldest_video_segment(
    db: Session,
    *,
    source_id: str,
    status: SegmentStatus,
    for_update: bool = False,
) -> VideoSegment | None:
    """Read the oldest segment for a source and status, without committing.

    Args:
        db: Session whose transaction is managed by the caller.
        source_id: Video source identifier.
        status: Processing state to filter by, using SegmentStatus.
        for_update: Lock the selected row and skip rows locked by other
            transactions. Defaults to False for a read-only selection.

    Returns:
        The matching segment with the earliest start_at_ms, using row ID to
        break ties, or None if no eligible row is available. When for_update
        is True, locked rows are excluded from selection.
    """
    statement = (
        select(VideoSegment)
        .where(
            VideoSegment.source_id == source_id,
            VideoSegment.status == status,
        )
        .order_by(
            VideoSegment.start_at_ms,
            VideoSegment.id,
        )
        .limit(1)
    )
    if for_update:
        statement = statement.with_for_update(skip_locked=True)

    return db.scalar(statement)


def update_video_segment_status(
    db: Session,
    segment: VideoSegment,
    *,
    status: SegmentStatus,
) -> None:
    """Update a segment's processing status without committing.

    Args:
        db: Session whose transaction is managed by the caller.
        segment: VideoSegment attached to db.
        status: New processing state, using SegmentStatus.

    Note:
        Flushes the change. The caller owns transition rules, locking, and commit.
    """
    segment.status = status
    db.flush()
