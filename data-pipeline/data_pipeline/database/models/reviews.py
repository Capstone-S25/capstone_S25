from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data_pipeline.database.base import Base

if TYPE_CHECKING:
    from data_pipeline.database.models.incidents import Incident


class ReviewOutcome(StrEnum):
    """A reviewer's decision on an incident."""

    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


class Review(Base):
    """A human review of an incident; at most one per incident.

    Severity is not stored here: it comes from the monitoring rule and
    reviewers cannot change it.

    Attributes:
        id: Auto-generated primary key.
        incident_id: Reviewed incident; deleted with it.
        outcome: ReviewOutcome decision.
        notes: Optional reviewer notes.
        corrected_type: Event type the reviewer says it really was, or None
            if the detected type was right.
        reviewer_id: User who submitted the review. Not a foreign key until
            the users table exists.
        created_at: Timezone-aware submission time, set by the database.
        updated_at: Time of the last edit, or None if never edited.
        updated_by: User who made the last edit, or None if never edited.
    """

    __tablename__ = "reviews"
    __table_args__ = (Index("ix_reviews_outcome", "outcome"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), unique=True
    )
    outcome: Mapped[ReviewOutcome] = mapped_column(
        Enum(
            ReviewOutcome,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="ck_reviews_outcome",
        )
    )
    notes: Mapped[str | None] = mapped_column(Text)
    corrected_type: Mapped[str | None] = mapped_column(String(64))
    reviewer_id: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[int | None]

    incident: Mapped["Incident"] = relationship(back_populates="review")
