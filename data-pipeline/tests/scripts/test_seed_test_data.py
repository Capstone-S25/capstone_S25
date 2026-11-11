from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from data_pipeline.database.models import (
    Detection,
    Incident,
    ReviewOutcome,
    Severity,
)
from scripts.seed_test_data import clear_seed_data, seed_test_data

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def test_seed_covers_every_filter_value(postgres_session):
    incidents = seed_test_data(postgres_session, now=NOW)

    assert len(incidents) == 30
    assert {i.location_id for i in incidents} == {1, 2, 3}
    assert len({i.camera_id for i in incidents}) == 4
    assert len({i.event_type for i in incidents}) == 3
    assert {i.severity for i in incidents} == set(Severity)
    assert all(NOW - timedelta(days=14) <= i.first_anomaly_at <= NOW for i in incidents)


def test_seed_detections_and_vlm_verdicts(postgres_session):
    incidents = seed_test_data(postgres_session, now=NOW)

    assert all(2 <= len(i.detections) <= 5 for i in incidents)
    assert all(len(i.vlm_results) == 1 for i in incidents)
    assert any(not d.is_flagged for i in incidents for d in i.detections)
    assert all(
        i.evidence_start_at <= d.detected_at <= i.evidence_end_at
        for i in incidents
        for d in i.detections
    )


def test_seed_reviews_and_settlements(postgres_session):
    incidents = seed_test_data(postgres_session, now=NOW)

    outcomes = Counter(i.review.outcome for i in incidents if i.review)
    assert sum(outcomes.values()) == 15
    assert set(outcomes) == set(ReviewOutcome)
    settled = [i for i in incidents if i.settlement]
    assert len(settled) == 2
    assert all(i.review.outcome == ReviewOutcome.ESCALATED for i in settled)


def test_rerunning_replaces_only_seed_data(postgres_session):
    seed_test_data(postgres_session, now=NOW)
    real = Incident(
        camera_id="camera_01",
        location_id=1,
        event_type="littering",
        severity=Severity.MINOR,
        first_anomaly_at=NOW,
        last_anomaly_at=NOW,
        evidence_start_at=NOW,
        evidence_end_at=NOW,
        evidence_clip_path="camera_01/real.mp4",
    )
    postgres_session.add(real)
    postgres_session.flush()

    assert clear_seed_data(postgres_session) == 30
    seed_test_data(postgres_session, now=NOW)

    assert postgres_session.scalar(select(func.count(Incident.id))) == 31
    assert postgres_session.get(Incident, real.id) is not None
    detections = postgres_session.scalar(select(func.count(Detection.id)))
    assert detections == sum(
        len(i.detections) for i in postgres_session.scalars(select(Incident))
    )
