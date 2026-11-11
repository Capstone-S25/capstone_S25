from datetime import timedelta

from data_pipeline.database.models import VLMAssessment, VLMResult


def url(incident_id):
    return f"/api/v1/incidents/{incident_id}"


def test_returns_full_reviewed_incident(client, seeded):
    incident = next(i for i in seeded if i.review)

    body = client.get(url(incident.id)).json()

    assert body["id"] == incident.id
    assert body["camera"] == {"id": incident.camera_id, "name": None}
    assert body["location"] == {"id": incident.location_id, "name": None}
    assert body["event_type"] == incident.event_type
    assert body["severity"] == incident.severity
    assert [d["id"] for d in body["detections"]] == [d.id for d in incident.detections]
    assert body["vlm_verdict"]["id"] == incident.vlm_results[0].id
    assert body["vlm_verdict"]["assessment"] == incident.vlm_results[0].assessment
    assert body["review"]["id"] == incident.review.id
    assert body["review"]["outcome"] == incident.review.outcome
    assert body["review"]["reviewer_id"] == incident.review.reviewer_id
    assert body["settlement"] is None
    assert body["edit_history"] == []


def test_unreviewed_incident_has_no_review(client, seeded):
    incident = next(i for i in seeded if i.review is None)

    body = client.get(url(incident.id)).json()

    assert body["review"] is None
    assert body["vlm_verdict"] is not None


def test_detections_include_unflagged_boxes_in_time_order(client, seeded):
    incident = next(i for i in seeded if not all(d.is_flagged for d in i.detections))

    detections = client.get(url(incident.id)).json()["detections"]

    assert [d["is_flagged"] for d in detections] == [
        d.is_flagged for d in incident.detections
    ]
    offsets = [d["offset_ms"] for d in detections]
    assert offsets == sorted(offsets)
    assert offsets[0] == 5000  # Clips start five seconds before the anomaly.


def test_returns_latest_vlm_verdict(client, seeded, postgres_session):
    incident = seeded[0]
    rerun = VLMResult(
        event_type=incident.event_type,
        assessment=VLMAssessment.UNCERTAIN,
        model_version="rerun",
        processed_at=incident.vlm_results[0].processed_at + timedelta(hours=1),
    )
    incident.vlm_results.append(rerun)
    postgres_session.flush()
    postgres_session.expire_all()

    verdict = client.get(url(incident.id)).json()["vlm_verdict"]

    assert verdict["id"] == rerun.id
    assert verdict["model_version"] == "rerun"


def test_missing_incident_returns_404(client, seeded):
    missing_id = max(i.id for i in seeded) + 1000

    response = client.get(url(missing_id))

    assert response.status_code == 404
    assert response.json() == {"detail": "Incident not found"}
