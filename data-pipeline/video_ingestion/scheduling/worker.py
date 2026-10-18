from sqlalchemy.orm import Session

from database.models.video_segments import SegmentStatus, VideoSegment
from database.repositories.video_segments import (
    get_oldest_video_segment,
    update_video_segment_status,
)


def claim_next_segment(db: Session, *, source_id: str) -> VideoSegment | None:
    """Lock the oldest available pending segment and mark it as processing.

    Args:
        db: Session whose transaction is managed by the caller.
        source_id: Video source whose next pending segment should be claimed.

    Returns:
        The oldest unlocked pending segment, with status changed to
        SegmentStatus.PROCESSING, or None if none is available for this source.
    """
    segment = get_oldest_video_segment(
        db, source_id=source_id, status=SegmentStatus.PENDING, for_update=True
    )
    if segment is None:
        return None

    update_video_segment_status(db, segment, status=SegmentStatus.PROCESSING)
    return segment
