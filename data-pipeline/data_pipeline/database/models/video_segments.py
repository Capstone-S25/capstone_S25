from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from data_pipeline.database.base import Base


class SegmentStatus(StrEnum):
    """Track a finalized segment's progress through video analysis.

    PENDING: Recorded and registered, waiting for a worker to claim it.
    PROCESSING: Claimed by a worker; decoding and analysis are in progress.
    PROCESSED: All required segment analysis finished successfully.
    FAILED: Processing could not finish because of an error.

    These states describe segment analysis, not recording or VLM event review.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class VideoSegment(Base):
    """Index one finalized clip for later searches across recording sessions.

    Video bytes remain on disk; only metadata lives in this table.

    Attributes:
        id: Auto-generated primary key.
        source_id: Video source identifier, e.g. "camera_01".
        session_id: Recording session UUID. Convert the recorder's hex string
            with UUID(session_id) when constructing a database row.
        path: File path relative to the recordings root, up to 1024 characters, e.g.
            "camera_01/<session_id>/000000.mp4".
        start_at_ms: Approximate segment start as UTC epoch milliseconds,
            calculated from the FFmpeg startup reference and CSV offset.
        end_at_ms: Approximate segment end as UTC epoch milliseconds,
            calculated from the same session reference.
        created_at: Timezone-aware metadata insertion time, set by the database.
            This is separate from the estimated video capture interval.
        status: Processing state; new segments start as pending.
        delete_at: Optional timezone-aware retention deadline. None means no
            deletion is scheduled.

    Note:
        UTC times are startup-based estimates, not camera capture times.
    """

    __tablename__ = "video_segments"
    __table_args__ = (
        CheckConstraint(
            "end_at_ms >= start_at_ms", name="ck_video_segments_time_order"
        ),
        UniqueConstraint("path", name="uq_video_segments_path"),
        Index("ix_video_segments_source_start", "source_id", "start_at_ms"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[str] = mapped_column(String(128))
    session_id: Mapped[UUID]
    path: Mapped[str] = mapped_column(String(1024))
    start_at_ms: Mapped[int] = mapped_column(BigInteger)
    end_at_ms: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[SegmentStatus] = mapped_column(
        Enum(
            SegmentStatus,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="ck_video_segments_status",
        ),
        server_default=SegmentStatus.PENDING.value,
    )
    delete_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
