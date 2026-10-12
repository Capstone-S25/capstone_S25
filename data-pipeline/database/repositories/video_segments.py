from pathlib import Path
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from database.models.video_segments import VideoSegment


def insert_video_segment(
    db: Session,
    *,
    source_id: str,
    session_id: str,
    storage_root: Path,
    path: Path,
    start_at_ms: int,
    end_at_ms: int,
) -> bool:
    """Insert segment metadata in PostgreSQL without committing.

    Args:
        db: Session whose transaction is managed by the caller.
        source_id: Video source identifier.
        session_id: Recording session UUID as a hex or hyphenated string.
        storage_root: Base directory for recorded files.
        path: Segment file path; resolved relative to storage_root for storage.
            Relative input paths are resolved against the working directory.
        start_at_ms: Approximate segment start as UTC epoch milliseconds.
        end_at_ms: Approximate segment end as UTC epoch milliseconds.

    Returns:
        True if inserted, or False if the normalized path already exists.
    """
    relative_path = path.resolve().relative_to(storage_root.resolve()).as_posix()

    statement = (
        insert(VideoSegment)
        .values(
            source_id=source_id,
            session_id=UUID(session_id),
            path=relative_path,
            start_at_ms=start_at_ms,
            end_at_ms=end_at_ms,
        )
        .on_conflict_do_nothing(constraint="uq_video_segments_path")
        .returning(VideoSegment.id)
    )

    return db.execute(statement).scalar_one_or_none() is not None
