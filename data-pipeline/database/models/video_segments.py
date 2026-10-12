from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


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
