from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from data_pipeline.api.schemas.reviews import Review
from data_pipeline.database import models
from data_pipeline.database.models import ReviewOutcome, Severity, VLMAssessment


class OutcomeFilter(StrEnum):
    """Review outcomes the incident list can filter by."""

    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"
    UNREVIEWED = "unreviewed"


class CameraRef(BaseModel):
    """A camera; name is None until the cameras table exists."""

    id: str
    name: str | None = None


class LocationRef(BaseModel):
    """A location; name is None until the locations table exists."""

    id: int | None
    name: str | None = None


class BoundingBox(BaseModel):
    """Box corners as fractions (0-1) of the frame width and height."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """One YOLO box; offset_ms is its time from the start of the clip."""

    id: int
    detected_at: datetime
    offset_ms: int
    class_label: str
    confidence: float
    bbox: BoundingBox
    is_flagged: bool

    @classmethod
    def from_model(
        cls, detection: models.Detection, clip_start: datetime
    ) -> "Detection":
        return cls(
            id=detection.id,
            detected_at=detection.detected_at,
            offset_ms=round(
                (detection.detected_at - clip_start).total_seconds() * 1000
            ),
            class_label=detection.class_label,
            confidence=detection.confidence,
            bbox=BoundingBox(
                x1=detection.bbox_x1,
                y1=detection.bbox_y1,
                x2=detection.bbox_x2,
                y2=detection.bbox_y2,
            ),
            is_flagged=detection.is_flagged,
        )


class VLMVerdict(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    assessment: VLMAssessment
    confidence: float | None
    description: str | None
    model_version: str
    processed_at: datetime


class IncidentListItem(BaseModel):
    id: int
    camera_id: str
    location_id: int | None
    event_type: str
    severity: Severity
    first_anomaly_at: datetime
    last_anomaly_at: datetime
    review_outcome: ReviewOutcome | None


class IncidentPage(BaseModel):
    items: list[IncidentListItem]
    total: int
    page: int
    page_size: int


class SimilarIncident(IncidentListItem):
    """A look-alike past incident; similarity runs from 1 (identical) down."""

    similarity: float


class VLMRerunRequest(BaseModel):
    """A waiting request for the VLM to check an incident again."""

    incident_id: int
    requested_at: datetime
    requested_by: int


class IncidentDetail(BaseModel):
    id: int
    camera: CameraRef
    location: LocationRef
    event_type: str
    severity: Severity
    first_anomaly_at: datetime
    last_anomaly_at: datetime
    evidence_start_at: datetime
    evidence_end_at: datetime
    created_at: datetime
    detections: list[Detection]
    vlm_verdict: VLMVerdict | None
    vlm_rerun_requested_at: datetime | None
    review: Review | None
    # TODO: Fill in with the settle (#25) and audit trail work.
    settlement: None = None
    edit_history: list[dict] = Field(default_factory=list)
