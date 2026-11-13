"""Record screened anomalies as incidents for the dashboard API.

The vision runtime journals every event transition. This listener mirrors the
ones a reviewer needs into PostgreSQL:

* a window the screening classifier calls an anomaly opens an incident, or
  extends the open incident of the same camera and event type;
* the evidence frames of that window are saved as a short MP4 and a thumbnail
  under ``<recordings>/evidence/<camera>/``;
* the screening result, and later Gemma's confirmation, are stored as
  ``vlm_results`` rows, so the API's "latest verdict" is Gemma's when it ran
  and the screening classifier's otherwise.

Every incident still goes to human review; nothing here raises an alert.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.database.models import (
    Detection,
    Incident,
    Severity,
    VLMAssessment,
    VLMResult,
)

logger = logging.getLogger(__name__)

# Monitoring rules will set severity per location; until they exist each event
# type has one default.
DEFAULT_SEVERITY = {
    "unattended_items": Severity.CRITICAL,
    "smoking": Severity.MODERATE,
    "rough_sleeping": Severity.MODERATE,
    "littering": Severity.MINOR,
    "loitering": Severity.MINOR,
}
SHORT_VIEW_TASKS = {"littering"}
ASSESSMENT = {
    "anomaly": VLMAssessment.CONFIRMED,
    "normal": VLMAssessment.REJECTED,
    "unknown": VLMAssessment.UNCERTAIN,
}


def confidence(prediction: dict) -> float | None:
    """Screening confidence in 0-1: the calibrated probability when the
    classifier supplies one, else a logistic of the margin over the threshold."""
    if "probability" in prediction:
        return min(1.0, max(0.0, float(prediction["probability"])))
    if "score" not in prediction or "threshold" not in prediction:
        return None
    margin = float(prediction["score"]) - float(prediction["threshold"])
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))


def write_clip(paths: list[str], times_s: list[float], target: Path) -> None:
    """Encode evidence frames as H.264 so a browser can play them."""
    import av
    from PIL import Image

    steps = [b - a for a, b in zip(times_s, times_s[1:]) if b > a]
    rate = max(1, min(30, round(1 / (sorted(steps)[len(steps) // 2])))) if steps else 1
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.mp4")
    with av.open(str(temporary), "w") as container:
        stream = None
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            if stream is None:
                width, height = image.size
                scale = min(1.0, 1280 / max(width, height))
                # H.264 needs even dimensions.
                size = (
                    max(2, round(width * scale) // 2 * 2),
                    max(2, round(height * scale) // 2 * 2),
                )
                stream = container.add_stream("libx264", rate=rate)
                stream.width, stream.height = size
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "26", "preset": "veryfast"}
            frame = av.VideoFrame.from_image(image.resize(size))
            for packet in stream.encode(frame):
                container.mux(packet)
        if stream is None:
            raise ValueError("No evidence frames to encode")
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(target)


def write_thumbnail(path: str, target: Path) -> None:
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((640, 640))
    image.save(target, format="JPEG", quality=85)


def flagged_box(candidate: dict, frame_path: str) -> tuple[float, ...] | None:
    """The rule's region as 0-1 fractions, or None for whole-scene reviews."""
    from PIL import Image

    box = candidate.get("roi_box")
    if not box or candidate.get("candidate_origin") == "scheduled_scene_review":
        return None
    with Image.open(frame_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = (float(v) for v in box)
    clip = lambda value: min(1.0, max(0.0, value))  # noqa: E731
    return (
        clip(min(x1, x2) / width),
        clip(min(y1, y2) / height),
        clip(max(x1, x2) / width),
        clip(max(y1, y2) / height),
    )


class IncidentRecorder:
    """EventStore listener: ``EventStore(..., on_write=IncidentRecorder(...))``.

    Args:
        session_factory: SQLAlchemy session factory for the application database.
        storage_root: Recordings root; evidence is written under ``evidence/``.
        quiet_seconds: A new anomaly window within this long of an incident's
            last one extends that incident instead of opening another (the
            planned ``t_quiet`` setting).
        location_id: Location to stamp on new incidents, when known.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage_root: Path,
        *,
        quiet_seconds: float = 30.0,
        location_id: int | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.storage_root = Path(storage_root)
        self.quiet = timedelta(seconds=quiet_seconds)
        self.location_id = location_id
        self.lock = threading.Lock()
        # (event id, task) -> incident id, for the later Gemma confirmation.
        self.incident_for: OrderedDict[tuple[str, str], int] = OrderedDict()

    def __call__(self, candidate: dict, row: dict) -> None:
        with self.lock:
            if row["status"] == "screened":
                self._screened(candidate, row.get("predictions") or {})
            elif row["status"] == "reviewed":
                self._reviewed(
                    candidate,
                    row.get("confirmation") or {},
                    (row.get("confirmation_details") or {}).get("model", "gemma"),
                )

    def _screened(self, candidate: dict, predictions: dict) -> None:
        for task, prediction in predictions.items():
            if prediction.get("decision") != "anomaly":
                continue
            view = candidate["views"]["short" if task in SHORT_VIEW_TASKS else "long"]
            paths, times = list(view["paths"]), [float(t) for t in view["times_s"]]
            at = datetime.fromtimestamp(candidate["timestamp_ms"] / 1000, UTC)
            start = at - timedelta(seconds=times[-1] - times[0])
            camera = str(candidate["source_id"])
            with self.session_factory.begin() as db:
                incident = db.scalars(
                    select(Incident)
                    .where(
                        Incident.camera_id == camera,
                        Incident.event_type == task,
                        Incident.last_anomaly_at >= start - self.quiet,
                        Incident.first_anomaly_at <= at,
                    )
                    .order_by(Incident.last_anomaly_at.desc())
                    .limit(1)
                    .with_for_update()
                ).first()
                created = incident is None
                if created:
                    incident = Incident(
                        camera_id=camera,
                        location_id=self.location_id,
                        event_type=task,
                        severity=DEFAULT_SEVERITY.get(task, Severity.MINOR),
                        first_anomaly_at=at,
                        last_anomaly_at=at,
                        evidence_start_at=start,
                        evidence_end_at=at,
                        evidence_clip_path="",
                    )
                    db.add(incident)
                    db.flush()
                    incident.evidence_clip_path = (
                        f"evidence/{camera}/incident_{incident.id:08d}.mp4"
                    )
                else:
                    incident.last_anomaly_at = max(incident.last_anomaly_at, at)
                    incident.evidence_end_at = max(incident.evidence_end_at, at)
                score = confidence(prediction)
                db.add(
                    VLMResult(
                        incident_id=incident.id,
                        event_type=task,
                        assessment=VLMAssessment.CONFIRMED,
                        confidence=score,
                        description=(
                            f"Screening classifier flagged {task.replace('_', ' ')} "
                            f"in the {times[-1] - times[0]:.0f} s window ending "
                            f"{at:%H:%M:%S} UTC (score {prediction.get('score', 0):.2f}, "
                            f"threshold {prediction.get('threshold', 0):.2f}). "
                            "Awaiting review."
                        ),
                        model_version=str(prediction.get("strategy", "screening")),
                        processed_at=datetime.now(UTC),
                    )
                )
                box = flagged_box(candidate, paths[-1])
                if box is not None:
                    db.add(
                        Detection(
                            incident_id=incident.id,
                            segment_id=candidate.get("segment_id"),
                            detected_at=at,
                            class_label=str(candidate.get("event_type", "region")),
                            confidence=score if score is not None else 0.0,
                            bbox_x1=box[0],
                            bbox_y1=box[1],
                            bbox_x2=box[2],
                            bbox_y2=box[3],
                            is_flagged=True,
                        )
                    )
                clip = self.storage_root / incident.evidence_clip_path
                incident_id = incident.id
            if created:
                try:
                    write_clip(paths, times, clip)
                    write_thumbnail(paths[-1], clip.with_suffix(".jpg"))
                except Exception:
                    logger.exception("Could not save evidence for %s", incident_id)
            self.incident_for[candidate["event_id"], task] = incident_id
            while len(self.incident_for) > 1024:
                self.incident_for.popitem(last=False)
            logger.info(
                "Incident %s | %s | %s | %s",
                incident_id,
                camera,
                task,
                "opened" if created else "extended",
            )

    def _reviewed(self, candidate: dict, confirmation: dict, model: str) -> None:
        for task, value in confirmation.items():
            incident_id = self.incident_for.get((candidate["event_id"], task))
            if incident_id is None:
                continue
            with self.session_factory.begin() as db:
                db.add(
                    VLMResult(
                        incident_id=incident_id,
                        event_type=task,
                        assessment=ASSESSMENT.get(
                            value.get("decision"), VLMAssessment.UNCERTAIN
                        ),
                        confidence=None,
                        description=value.get("evidence"),
                        model_version=model,
                        processed_at=datetime.now(UTC),
                    )
                )
