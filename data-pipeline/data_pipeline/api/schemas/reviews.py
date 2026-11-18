from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_pipeline.database.models import ReviewOutcome, Severity, VLMAssessment


class Review(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    incident_id: int
    outcome: ReviewOutcome
    notes: str | None
    corrected_type: str | None
    reviewer_id: int
    created_at: datetime
    updated_at: datetime | None
    updated_by: int | None


class ReviewCreate(BaseModel):
    """Body for submitting a review (#22); severity cannot be set here."""

    model_config = ConfigDict(extra="forbid")

    outcome: ReviewOutcome
    notes: str | None = Field(default=None, max_length=5000)
    corrected_type: str | None = Field(default=None, min_length=1, max_length=64)


class ReviewUpdate(BaseModel):
    """Body for editing a review (#23); only the fields sent are changed.

    Send notes or corrected_type as null to clear them.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: ReviewOutcome | None = None
    notes: str | None = Field(default=None, max_length=5000)
    corrected_type: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_changes(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("Send at least one of outcome, notes, corrected_type")
        if "outcome" in self.model_fields_set and self.outcome is None:
            raise ValueError("outcome cannot be null")
        return self


class VLMHint(BaseModel):
    """The VLM's verdict, shown to reviewers as a hint."""

    model_config = ConfigDict(from_attributes=True)

    assessment: VLMAssessment
    confidence: float | None
    description: str | None


class ReviewQueueItem(BaseModel):
    id: int
    camera_id: str
    location_id: int | None
    event_type: str
    severity: Severity
    first_anomaly_at: datetime
    last_anomaly_at: datetime
    vlm_hint: VLMHint | None


class ReviewQueuePage(BaseModel):
    items: list[ReviewQueueItem]
    total: int
    page: int
    page_size: int
