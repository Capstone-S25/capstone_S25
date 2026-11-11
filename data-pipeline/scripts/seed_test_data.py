"""Fill the database with fake incidents for building and testing the dashboard.

Run from data-pipeline/ with ``npm run db:seed``. Running it again replaces
the previous seed data; incidents not created by this script are untouched.

Seeded incidents are recognisable by an evidence_clip_path starting with
"seed/". Locations and cameras are plain IDs until Person A's tables exist:

    location 1 "Main Gate":  camera_01, camera_02
    location 2 "Car Park":   camera_03
    location 3 "Food Court": camera_04
"""

import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import delete
from sqlalchemy.orm import Session

from data_pipeline.database.models import (
    Detection,
    Incident,
    Review,
    ReviewOutcome,
    Settlement,
    SettlementOutcome,
    Severity,
    VLMAssessment,
    VLMResult,
)
from data_pipeline.database.session import create_engine_and_session_factory

SEED_PREFIX = "seed/"
INCIDENT_COUNT = 30
CAMERAS = [("camera_01", 1), ("camera_02", 1), ("camera_03", 2), ("camera_04", 3)]
EVENT_TYPES = ["littering", "loitering", "unattended_item"]
FLAGGED_LABELS = {
    "littering": ["person", "bottle", "cup"],
    "loitering": ["person"],
    "unattended_item": ["backpack", "suitcase", "handbag"],
}
VLM_DESCRIPTIONS = {
    VLMAssessment.CONFIRMED: "The clip shows {event}.",
    VLMAssessment.REJECTED: "No {event} is visible; likely a false trigger.",
    VLMAssessment.UNCERTAIN: "Possible {event}, but the view is partly blocked.",
}
EVENT_PHRASES = {
    "littering": "a person dropping an item and walking away",
    "loitering": "a person staying in the area for a long time",
    "unattended_item": "a bag left with nobody nearby",
}
REVIEW_OUTCOMES = list(ReviewOutcome)
SEVERITIES = list(Severity)
CLIP_PADDING = timedelta(seconds=5)


def clear_seed_data(db: Session) -> int:
    """Delete seeded incidents and everything attached to them.

    Returns:
        Number of incidents deleted.
    """
    result = db.execute(
        delete(Incident).where(Incident.evidence_clip_path.startswith(SEED_PREFIX))
    )
    return result.rowcount


def seed_test_data(
    db: Session, *, now: datetime | None = None, seed: int = 7
) -> list[Incident]:
    """Add seeded incidents without committing.

    Every location, camera, event type and severity appears. About half
    the incidents are reviewed with a mix of outcomes, and two escalated
    ones are settled. Each incident has 2-5 detections; when there are
    three or more, the last is an unflagged passer-by.

    Args:
        db: Session whose transaction is managed by the caller.
        now: Latest possible incident time; defaults to the current time.
        seed: Random seed, so the same arguments give the same data.

    Returns:
        The new incidents, flushed so their IDs are set, with detections,
        vlm_results, review and settlement loaded.
    """
    rng = random.Random(seed)
    now = now or datetime.now(UTC)
    reviewed = sorted(rng.sample(range(INCIDENT_COUNT), INCIDENT_COUNT // 2))
    incidents = []
    settled = 0
    for i in range(INCIDENT_COUNT):
        camera_id, location_id = CAMERAS[i % len(CAMERAS)]
        event_type = EVENT_TYPES[i % len(EVENT_TYPES)]
        first_at = now - timedelta(seconds=rng.uniform(3600, 14 * 86400 - 3600))
        last_at = first_at + timedelta(seconds=rng.uniform(5, 40))
        incident = Incident(
            camera_id=camera_id,
            location_id=location_id,
            event_type=event_type,
            severity=SEVERITIES[(i // len(EVENT_TYPES)) % len(SEVERITIES)],
            first_anomaly_at=first_at,
            last_anomaly_at=last_at,
            evidence_start_at=first_at - CLIP_PADDING,
            evidence_end_at=last_at + CLIP_PADDING,
            evidence_clip_path=f"{SEED_PREFIX}{camera_id}/incident_{i:02d}.mp4",
        )
        incident.detections = _detections(rng, event_type, first_at, last_at)

        if i % 5 == 4:
            assessment = VLMAssessment.REJECTED
        elif i % 7 == 6:
            assessment = VLMAssessment.UNCERTAIN
        else:
            assessment = VLMAssessment.CONFIRMED
        processed_at = incident.evidence_end_at + timedelta(seconds=rng.uniform(10, 90))
        incident.vlm_results = [
            VLMResult(
                event_type=event_type,
                assessment=assessment,
                confidence=round(rng.uniform(0.55, 0.98), 3),
                description=VLM_DESCRIPTIONS[assessment].format(
                    event=EVENT_PHRASES[event_type]
                ),
                model_version="seed-vlm-1",
                processed_at=processed_at,
            )
        ]

        if i in reviewed:
            k = reviewed.index(i)
            outcome = REVIEW_OUTCOMES[k % len(REVIEW_OUTCOMES)]
            created_at = min(processed_at + timedelta(minutes=rng.uniform(5, 600)), now)
            edited = k % 5 == 2
            incident.review = Review(
                outcome=outcome,
                notes=f"Seed review {k + 1}: {outcome} after watching the clip.",
                corrected_type=(
                    EVENT_TYPES[(i + 1) % len(EVENT_TYPES)] if k % 4 == 1 else None
                ),
                reviewer_id=1 if k % 2 == 0 else 2,
                created_at=created_at,
                updated_at=created_at + timedelta(minutes=30) if edited else None,
                updated_by=2 if edited else None,
            )
            if outcome == ReviewOutcome.ESCALATED and settled < 2:
                incident.settlement = Settlement(
                    outcome=list(SettlementOutcome)[settled],
                    notes="Seed settlement after admin follow-up.",
                    settled_by=1,
                    created_at=created_at + timedelta(hours=2),
                )
                settled += 1
        incidents.append(incident)

    db.add_all(incidents)
    db.flush()
    return incidents


def _detections(
    rng: random.Random, event_type: str, first_at: datetime, last_at: datetime
) -> list[Detection]:
    count = rng.randint(2, 5)
    span = (last_at - first_at).total_seconds()
    offsets = sorted([0.0, *(rng.uniform(0, span) for _ in range(count - 1))])
    detections = []
    for j, offset in enumerate(offsets):
        flagged = not (count >= 3 and j == count - 1)
        x1, y1 = rng.uniform(0, 0.7), rng.uniform(0, 0.6)
        detections.append(
            Detection(
                detected_at=first_at + timedelta(seconds=offset),
                class_label=(
                    rng.choice(FLAGGED_LABELS[event_type]) if flagged else "person"
                ),
                confidence=round(rng.uniform(0.35, 0.95), 3),
                bbox_x1=round(x1, 4),
                bbox_y1=round(y1, 4),
                bbox_x2=round(x1 + rng.uniform(0.05, 0.25), 4),
                bbox_y2=round(y1 + rng.uniform(0.1, 0.35), 4),
                is_flagged=flagged,
            )
        )
    return detections


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL in data-pipeline/.env or your environment")
    engine, session_factory = create_engine_and_session_factory(database_url)
    try:
        with session_factory.begin() as db:
            removed = clear_seed_data(db)
            incidents = seed_test_data(db)
            reviewed = sum(incident.review is not None for incident in incidents)
            settled = sum(incident.settlement is not None for incident in incidents)
    finally:
        engine.dispose()
    print(
        f"Removed {removed} old seed incidents. Added {len(incidents)} incidents:"
        f" {reviewed} reviewed ({settled} settled), {len(incidents) - reviewed}"
        " unreviewed."
    )


if __name__ == "__main__":
    main()
