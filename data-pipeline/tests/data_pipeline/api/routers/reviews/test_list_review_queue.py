import pytest
from sqlalchemy import delete

from data_pipeline.database.models import VLMResult

URL = "/api/v1/reviews/queue"
SEVERITY_ORDER = {"critical": 0, "moderate": 1, "minor": 2}


def queue_order(incidents):
    """Expected queue: unreviewed, critical first, then longest waiting."""
    waiting = [i for i in incidents if i.review is None]
    return [
        i.id
        for i in sorted(
            waiting,
            key=lambda i: (SEVERITY_ORDER[i.severity], i.first_anomaly_at, i.id),
        )
    ]


def get_ids(client, **params):
    response = client.get(URL, params={"page_size": 100, **params})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == len(body["items"])
    return [item["id"] for item in body["items"]]


def test_lists_unreviewed_incidents_most_urgent_first(client, seeded):
    expected = queue_order(seeded)

    assert len(expected) == 15
    assert get_ids(client) == expected


def test_items_carry_the_vlm_verdict_as_a_hint(client, seeded):
    incident = next(i for i in seeded if i.review is None)
    vlm = incident.vlm_results[-1]

    items = client.get(URL, params={"page_size": 100}).json()["items"]

    item = next(item for item in items if item["id"] == incident.id)
    assert item["vlm_hint"] == {
        "assessment": vlm.assessment,
        "confidence": vlm.confidence,
        "description": vlm.description,
    }
    assert item["severity"] == incident.severity
    assert item["camera_id"] == incident.camera_id


def test_incident_without_vlm_result_has_no_hint(client, seeded, postgres_session):
    incident = next(i for i in seeded if i.review is None)
    postgres_session.execute(
        delete(VLMResult).where(VLMResult.incident_id == incident.id)
    )

    items = client.get(URL, params={"page_size": 100}).json()["items"]

    assert next(i for i in items if i["id"] == incident.id)["vlm_hint"] is None


@pytest.mark.parametrize(
    ("params", "matches"),
    [
        ({"location_id": 1}, lambda i: i.location_id == 1),
        ({"camera_id": "camera_04"}, lambda i: i.camera_id == "camera_04"),
        ({"event_type": "littering"}, lambda i: i.event_type == "littering"),
        ({"severity": "critical"}, lambda i: i.severity == "critical"),
        ({"severity": "minor"}, lambda i: i.severity == "minor"),
    ],
)
def test_filters(client, seeded, params, matches):
    expected = queue_order([i for i in seeded if matches(i)])

    assert expected, "seed data should include matches for this filter"
    assert get_ids(client, **params) == expected


def test_paginates_with_total(client, seeded):
    expected = queue_order(seeded)

    first = client.get(URL, params={"page_size": 10}).json()
    second = client.get(URL, params={"page_size": 10, "page": 2}).json()

    assert first["total"] == second["total"] == 15
    assert [i["id"] for i in first["items"] + second["items"]] == expected


def test_reviewed_incident_leaves_the_queue(client, seeded):
    incident_id = queue_order(seeded)[0]

    client.post(f"/api/v1/incidents/{incident_id}/review", json={"outcome": "accepted"})

    assert incident_id not in get_ids(client)


@pytest.mark.parametrize(
    "params", [{"page": 0}, {"page_size": 101}, {"severity": "high"}]
)
def test_rejects_invalid_parameters(client, params):
    assert client.get(URL, params=params).status_code == 422
