import pytest


def url(incident_id):
    return f"/api/v1/incidents/{incident_id}/review"


def unreviewed(seeded):
    return next(i for i in seeded if i.review is None)


def test_creates_the_review(client, seeded, account):
    incident = unreviewed(seeded)

    response = client.post(
        url(incident.id),
        json={
            "outcome": "dismissed",
            "notes": "Just a delivery driver.",
            "corrected_type": "loitering",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["incident_id"] == incident.id
    assert body["outcome"] == "dismissed"
    assert body["notes"] == "Just a delivery driver."
    assert body["corrected_type"] == "loitering"
    assert body["reviewer_id"] == account.user_id
    assert body["created_at"] is not None
    assert body["updated_at"] is None
    assert body["updated_by"] is None


def test_notes_and_corrected_type_are_optional(client, seeded):
    response = client.post(url(unreviewed(seeded).id), json={"outcome": "accepted"})

    assert response.status_code == 201
    assert response.json()["notes"] is None
    assert response.json()["corrected_type"] is None


def test_review_shows_on_the_incident(client, seeded):
    incident = unreviewed(seeded)

    review = client.post(url(incident.id), json={"outcome": "escalated"}).json()

    assert client.get(f"/api/v1/incidents/{incident.id}").json()["review"] == review


def test_second_review_is_rejected(client, seeded):
    incident = next(i for i in seeded if i.review is not None)

    response = client.post(url(incident.id), json={"outcome": "accepted"})

    assert response.status_code == 409
    detail = client.get(f"/api/v1/incidents/{incident.id}").json()["review"]
    assert detail["outcome"] == incident.review.outcome


def test_missing_incident_returns_404(client, seeded):
    response = client.post(
        url(max(i.id for i in seeded) + 1000), json={"outcome": "accepted"}
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"outcome": "approved"},
        {"outcome": "accepted", "severity": "critical"},
        {"outcome": "accepted", "corrected_type": ""},
        {"outcome": "accepted", "notes": "x" * 5001},
    ],
)
def test_rejects_invalid_bodies(client, seeded, body):
    response = client.post(url(unreviewed(seeded).id), json=body)

    assert response.status_code == 422
