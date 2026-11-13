import os
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from data_pipeline.api.app import create_app
from data_pipeline.api.database import get_db
from data_pipeline.database.models import Incident, Severity, VLMAssessment
from data_pipeline.detection.incidents import IncidentRecorder, confidence

START_MS = int(datetime(2026, 11, 13, 9, 0, tzinfo=UTC).timestamp() * 1000)


@pytest.fixture
def recorder(postgres_session, tmp_path):
    factory = sessionmaker(
        bind=postgres_session.connection(), join_transaction_mode="create_savepoint"
    )
    return IncidentRecorder(factory, tmp_path / "recordings", quiet_seconds=30)


def candidate(tmp_path, name, at_s, *, scene=False):
    paths = []
    for index in range(4):
        path = tmp_path / f"{name}_{index}.jpg"
        Image.new("RGB", (321, 241), (40 * index, 80, 120)).save(path)
        paths.append(str(path))
    times = [at_s - 20 + index * 20 / 3 for index in range(4)]
    view = {"paths": paths, "times_s": times}
    return {
        "event_id": name,
        "event_type": "periodic_review" if scene else "unattended_object",
        "candidate_origin": "scheduled_scene_review" if scene else "rule",
        "source_id": "camera_01",
        "segment_id": None,
        "timestamp_s": at_s,
        "timestamp_ms": START_MS + round(at_s * 1000),
        "roi_box": [32.1, 24.1, 160.5, 120.5],
        "views": {"short": view, "long": view},
        "tasks": ["unattended_items", "smoking"],
    }


def screened(decision, score=1.4):
    return {
        "status": "screened",
        "predictions": {
            "unattended_items": {
                "decision": decision,
                "score": score,
                "threshold": 0.4,
                "strategy": "ensemble-20261111",
            },
            "smoking": {"decision": "normal", "score": -1.0, "threshold": 0.2},
        },
    }


def test_confidence_prefers_calibrated_probability():
    assert confidence({"probability": 0.83, "score": 0, "threshold": 5}) == 0.83
    assert confidence({"score": 1.0, "threshold": 1.0}) == 0.5
    assert confidence({"decision": "unknown"}) is None


def test_anomaly_window_opens_extends_and_records_confirmation(
    postgres_session, recorder, tmp_path
):
    first = candidate(tmp_path, "event_a", 40.0)
    recorder(first, screened("anomaly"))
    recorder(candidate(tmp_path, "event_b", 50.0), screened("anomaly"))
    recorder(candidate(tmp_path, "event_c", 60.0), screened("normal"))
    recorder(
        first,
        {
            "status": "reviewed",
            "confirmation": {
                "unattended_items": {"decision": "anomaly", "evidence": "Bag left."},
                "smoking": {"decision": "normal", "evidence": "No cigarette."},
            },
            "confirmation_details": {"model": "gemma4-26b-a4b"},
        },
    )

    incidents = list(postgres_session.scalars(select(Incident)))
    assert len(incidents) == 1
    incident = incidents[0]
    assert incident.event_type == "unattended_items"
    assert incident.severity == Severity.CRITICAL
    assert (incident.last_anomaly_at - incident.first_anomaly_at).total_seconds() == 10
    assert (
        incident.first_anomaly_at - incident.evidence_start_at
    ).total_seconds() == 20
    clip = recorder.storage_root / incident.evidence_clip_path
    assert clip.stat().st_size > 0 and clip.with_suffix(".jpg").is_file()

    postgres_session.refresh(incident)
    verdicts = incident.vlm_results
    assert [v.model_version for v in verdicts] == [
        "ensemble-20261111",
        "ensemble-20261111",
        "gemma4-26b-a4b",
    ]
    assert verdicts[-1].assessment == VLMAssessment.CONFIRMED
    assert verdicts[-1].description == "Bag left."
    assert 0.5 < verdicts[0].confidence < 1
    box = incident.detections[0]
    assert (box.bbox_x1, box.bbox_y1) == pytest.approx((0.1, 0.1))
    assert (box.bbox_x2, box.bbox_y2) == pytest.approx((0.5, 0.5))


def test_gap_longer_than_quiet_time_opens_a_new_incident(
    postgres_session, recorder, tmp_path
):
    recorder(candidate(tmp_path, "early", 40.0, scene=True), screened("anomaly"))
    recorder(candidate(tmp_path, "late", 140.0, scene=True), screened("anomaly"))
    incidents = list(postgres_session.scalars(select(Incident)))
    assert len(incidents) == 2
    # Whole-scene reviews have no flagged region to draw.
    assert all(not incident.detections for incident in incidents)


def test_evidence_endpoints_serve_only_files_under_the_recordings_root(
    postgres_session, recorder, tmp_path
):
    recorder(candidate(tmp_path, "event_a", 40.0), screened("anomaly"))
    incident = postgres_session.scalars(select(Incident)).one()
    app = create_app(os.environ["TEST_DATABASE_URL"], recorder.storage_root)

    def override_get_db():
        yield postgres_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        clip = client.get(f"/api/v1/incidents/{incident.id}/clip")
        assert clip.status_code == 200
        assert clip.headers["content-type"] == "video/mp4"
        thumbnail = client.get(f"/api/v1/incidents/{incident.id}/thumbnail")
        assert thumbnail.headers["content-type"] == "image/jpeg"
        assert client.get("/api/v1/incidents/999999/clip").status_code == 404

        outside = tmp_path / "secret.mp4"
        outside.write_bytes(b"not evidence")
        incident.evidence_clip_path = "../secret.mp4"
        postgres_session.flush()
        assert client.get(f"/api/v1/incidents/{incident.id}/clip").status_code == 404
