def url(incident_id):
    return f"/api/v1/incidents/{incident_id}/detections"


def test_returns_only_flagged_boxes_in_time_order(client, seeded):
    incident = next(i for i in seeded if not all(d.is_flagged for d in i.detections))
    flagged = [d for d in incident.detections if d.is_flagged]

    body = client.get(url(incident.id)).json()

    assert [d["id"] for d in body] == [d.id for d in flagged]
    assert all(d["is_flagged"] for d in body)
    times = [d["detected_at"] for d in body]
    assert times == sorted(times)


def test_box_fields(client, seeded):
    incident = seeded[0]
    detection = incident.detections[0]

    first = client.get(url(incident.id)).json()[0]

    assert first["class_label"] == detection.class_label
    assert first["confidence"] == detection.confidence
    assert first["bbox"] == {
        "x1": detection.bbox_x1,
        "y1": detection.bbox_y1,
        "x2": detection.bbox_x2,
        "y2": detection.bbox_y2,
    }
    assert first["offset_ms"] == 5000


def test_incident_without_flagged_boxes_returns_empty_list(
    client, seeded, postgres_session
):
    incident = seeded[0]
    for detection in incident.detections:
        detection.is_flagged = False
    postgres_session.flush()

    response = client.get(url(incident.id))

    assert response.status_code == 200
    assert response.json() == []


def test_missing_incident_returns_404(client, seeded):
    missing_id = max(i.id for i in seeded) + 1000

    response = client.get(url(missing_id))

    assert response.status_code == 404
    assert response.json() == {"detail": "Incident not found"}
