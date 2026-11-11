from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data_pipeline.database.base import Base

if TYPE_CHECKING:
    from data_pipeline.database.models.incidents import Incident


class VLMAssessment(StrEnum):
    """The VLM's verdict on whether the YOLO candidate is a real incident."""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class VLMResult(Base):
    """One VLM verification of an incident's evidence clip.

    An incident can have several results if it is re-run; the latest
    ``processed_at`` is the current verdict.

    Attributes:
        id: Auto-generated primary key.
        incident_id: Incident that was verified; deleted with it.
        event_type: Event type the VLM judged, e.g. "littering".
        assessment: VLMAssessment verdict.
        confidence: Optional VLM confidence from 0 to 1.
        description: Optional free-text explanation from the VLM.
        model_version: Model name and version that produced the verdict.
        processed_at: When the VLM finished.

    Note:
        The pgvector ``embedding`` column is added with similar incidents.
    """

    __tablename__ = "vlm_results"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_vlm_results_confidence"
        ),
        Index("ix_vlm_results_incident_processed_at", "incident_id", "processed_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE")
    )
    event_type: Mapped[str] = mapped_column(String(64))
    assessment: Mapped[VLMAssessment] = mapped_column(
        Enum(
            VLMAssessment,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="ck_vlm_results_assessment",
        )
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    description: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(128))
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    incident: Mapped["Incident"] = relationship(back_populates="vlm_results")
