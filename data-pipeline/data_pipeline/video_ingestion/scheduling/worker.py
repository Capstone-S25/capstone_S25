from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.database.models.video_segments import SegmentStatus, VideoSegment
from data_pipeline.database.repositories.video_segments import (
    get_oldest_video_segment,
    update_video_segment_status,
)


@dataclass(frozen=True)
class SegmentJob:
    """Describe a claimed video segment ready for processing.

    Attributes:
        segment_id: Segment's database ID, used to update its processing status.
        source_id: Identifier of the camera or file source that recorded it.
        session_id: UUID of the recording session containing the segment.
        path: Video file path relative to the recordings root, e.g.
            "camera_01/<session_id>/000000.mp4".
        start_at_ms: Approximate segment start as UTC epoch milliseconds.
        end_at_ms: Approximate segment end as UTC epoch milliseconds.
    """

    segment_id: int
    source_id: str
    session_id: UUID
    path: str
    start_at_ms: int
    end_at_ms: int


def _claim_next_segment(db: Session, *, source_id: str) -> VideoSegment | None:
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


class SegmentScheduler:
    """Claim pending video segments across cameras in round-robin order.

    Notes:
        Reuse the instance to preserve its camera cursor between calls.
        Each instance is intended for one caller.
    """

    def __init__(
        self,
        *,
        source_ids: Sequence[str],
        session_factory: sessionmaker[Session],
    ) -> None:
        """
        Args:
            source_ids: Camera IDs to check for pending work. Duplicates are
                removed while preserving their first occurrence order.
            session_factory: Factory for short-lived claiming transactions.
        """
        self._source_ids = tuple(dict.fromkeys(source_ids))
        self._session_factory = session_factory
        self._cursor = 0

    def get_next_job(self) -> SegmentJob | None:
        """Get the next video segment ready for processing.

        Takes turns checking cameras, skipping those without pending segments.
        Each call checks every camera at most once. The next call resumes from
        the following camera so busy cameras do not monopolize processing.

        Returns:
            A SegmentJob for the chosen camera's oldest available pending
            segment, marked PROCESSING so other workers do not claim it again.
            Returns None if there are no cameras or none has available work.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database cannot select or
                reserve a segment.
        """
        for _ in self._source_ids:
            source_id = self._source_ids[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._source_ids)

            job = None
            with self._session_factory.begin() as db:
                segment = _claim_next_segment(db, source_id=source_id)
                if segment is not None:
                    job = SegmentJob(
                        segment_id=segment.id,
                        source_id=segment.source_id,
                        session_id=segment.session_id,
                        path=segment.path,
                        start_at_ms=segment.start_at_ms,
                        end_at_ms=segment.end_at_ms,
                    )

            if job is not None:
                return job
        return None
