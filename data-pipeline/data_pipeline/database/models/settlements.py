from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data_pipeline.database.base import Base

if TYPE_CHECKING:
    from data_pipeline.database.models.incidents import Incident


class SettlementOutcome(StrEnum):
    """How an admin closed an escalated incident."""

    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class Settlement(Base):
    """An admin's settlement of an escalated incident; at most one per incident.

    Attributes:
        id: Auto-generated primary key.
        incident_id: Settled incident; deleted with it.
        outcome: SettlementOutcome decision.
        notes: Optional admin notes.
        settled_by: Admin who settled it. Not a foreign key until the users
            table exists.
        created_at: Timezone-aware settlement time, set by the database.
        updated_at: Time of the last edit, or None if never edited.
        updated_by: User who made the last edit, or None if never edited.
    """

    __tablename__ = "settlements"

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), unique=True
    )
    outcome: Mapped[SettlementOutcome] = mapped_column(
        Enum(
            SettlementOutcome,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="ck_settlements_outcome",
        )
    )
    notes: Mapped[str | None] = mapped_column(Text)
    settled_by: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[int | None]

    incident: Mapped["Incident"] = relationship(back_populates="settlement")
