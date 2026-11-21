"""Build the cards for one dashboard page in a bounded number of queries."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import case, func, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session

from data_pipeline.database.models import Incident, Review, VLMAssessment, VLMResult


class CardVerdict(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    assessment: VLMAssessment
    confidence: float | None
    description: str | None
    model_version: str


class IncidentCard(BaseModel):
    evidence_start_at: datetime
    review_notes: str | None = None
    screening: CardVerdict | None = None
    confirmation: CardVerdict | None = None
    review_count: int = 0
    confirmation_count: int = 0


def incident_cards(db: Session, incidents: list[Incident], *, include_notes: bool):
    cards = {
        row.id: IncidentCard(evidence_start_at=row.evidence_start_at)
        for row in incidents
    }
    if not cards:
        return cards
    # Match the dashboard's prior policy: show the latest confirming review,
    # falling back to the latest other review, plus the latest screening score.
    kind = case(
        (VLMResult.confidence.is_not(None), "screening"),
        (VLMResult.assessment == VLMAssessment.CONFIRMED, "confirmed"),
        else_="other",
    )
    rows = db.execute(
        select(
            VLMResult.incident_id,
            VLMResult.assessment,
            VLMResult.confidence,
            VLMResult.description,
            VLMResult.model_version,
            kind.label("kind"),
            func.count()
            .over(partition_by=(VLMResult.incident_id, kind))
            .label("kind_count"),
        )
        .where(VLMResult.incident_id.in_(cards))
        .ext(distinct_on(VLMResult.incident_id, kind))
        .order_by(
            VLMResult.incident_id,
            kind,
            VLMResult.processed_at.desc(),
            VLMResult.id.desc(),
        )
    )
    for row in rows:
        card = cards[row.incident_id]
        verdict = CardVerdict.model_validate(row)
        if row.kind == "screening":
            card.screening = verdict
        else:
            card.review_count += row.kind_count
            if row.kind == "confirmed":
                card.confirmation = verdict
                card.confirmation_count = row.kind_count
            elif card.confirmation is None:
                card.confirmation = verdict
    if include_notes:
        for incident_id, notes in db.execute(
            select(Review.incident_id, Review.notes).where(
                Review.incident_id.in_(cards)
            )
        ):
            cards[incident_id].review_notes = notes
    return cards
