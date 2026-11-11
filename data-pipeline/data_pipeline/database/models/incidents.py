from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data_pipeline.database.base import Base

if TYPE_CHECKING:
    from data_pipeline.database.models.detections import Detection
    from data_pipeline.database.models.reviews import Review
    from data_pipeline.database.models.settlements import Settlement
    from data_pipeline.database.models.vlm_results import VLMResult


class Severity(StrEnum):
    """How urgent an incident is; copied from the monitoring rule that fired."""

    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class Incident(Base):
    """One anomaly worth a human look, with its evidence clip.

    Planned as ``anomaly_events`` in the original schema and renamed to
    ``incidents`` in the API plan.

    Attributes:
        id: Auto-generated primary key.
        camera_id: Camera identifier, the same value as
            ``video_segments.source_id``, e.g. "camera_01". Not a foreign key
            until the cameras table exists.
        location_id: Location the camera belonged to when the incident
            happened. Not a foreign key until the locations table exists.
        event_type: Event type name, e.g. "littering". Becomes a foreign key
            once the event_types table exists.
        severity: Severity copied from the triggering monitoring rule.
        first_anomaly_at: When the anomaly was first detected; incidents are
            listed and date-filtered by this time.
        last_anomaly_at: When the anomaly was last detected.
        evidence_start_at: Start of the saved evidence clip.
        evidence_end_at: End of the saved evidence clip.
        evidence_clip_path: Clip path relative to the recordings root (later
            an S3 key), up to 1024 characters.
        created_at: Timezone-aware insertion time, set by the database.
    """

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_first_anomaly_at_id", "first_anomaly_at", "id"),
        Index("ix_incidents_camera_id", "camera_id"),
        Index("ix_incidents_location_id", "location_id"),
        Index("ix_incidents_event_type", "event_type"),
        Index("ix_incidents_severity", "severity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(128))
    location_id: Mapped[int | None]
    event_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[Severity] = mapped_column(
        Enum(
            Severity,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="ck_incidents_severity",
        )
    )
    first_anomaly_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_anomaly_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evidence_start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evidence_end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evidence_clip_path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="incident", order_by="Detection.detected_at"
    )
    vlm_results: Mapped[list["VLMResult"]] = relationship(
        back_populates="incident", order_by="VLMResult.processed_at"
    )
    review: Mapped["Review | None"] = relationship(back_populates="incident")
    settlement: Mapped["Settlement | None"] = relationship(back_populates="incident")
