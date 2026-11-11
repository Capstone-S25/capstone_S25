from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data_pipeline.database.base import Base

if TYPE_CHECKING:
    from data_pipeline.database.models.incidents import Incident


class Detection(Base):
    """One YOLO bounding box seen during an incident.

    Box corners are normalized to 0-1 of the frame width and height (YOLO's
    ``xyxyn``), so the dashboard can draw them over a clip at any size.

    Attributes:
        id: Auto-generated primary key.
        incident_id: Incident the box belongs to; deleted with it.
        segment_id: Video segment the frame came from. Set to None when
            retention deletes the segment.
        detected_at: Capture time of the frame.
        class_label: YOLO class name, e.g. "person" or "backpack".
        confidence: YOLO confidence from 0 to 1.
        bbox_x1: Left edge.
        bbox_y1: Top edge.
        bbox_x2: Right edge.
        bbox_y2: Bottom edge.
        is_flagged: True when this box triggered the anomaly; only flagged
            boxes are drawn during playback.
    """

    __tablename__ = "detections"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_detections_confidence"
        ),
        CheckConstraint(
            "bbox_x1 >= 0 AND bbox_y1 >= 0 AND bbox_x2 <= 1 AND bbox_y2 <= 1"
            " AND bbox_x1 <= bbox_x2 AND bbox_y1 <= bbox_y2",
            name="ck_detections_bbox",
        ),
        Index("ix_detections_incident_detected_at", "incident_id", "detected_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )
    segment_id: Mapped[int | None] = mapped_column(
        ForeignKey("video_segments.id", ondelete="SET NULL")
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    class_label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x1: Mapped[float] = mapped_column(Float)
    bbox_y1: Mapped[float] = mapped_column(Float)
    bbox_x2: Mapped[float] = mapped_column(Float)
    bbox_y2: Mapped[float] = mapped_column(Float)
    is_flagged: Mapped[bool] = mapped_column(server_default=true())

    incident: Mapped["Incident"] = relationship(back_populates="detections")
