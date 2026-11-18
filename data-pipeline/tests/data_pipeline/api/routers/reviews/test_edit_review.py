import pytest


def url(review_id):
    return f"/api/v1/reviews/{review_id}"


def unsettled_review(seeded):
    return next(i.review for i in seeded if i.review and i.settlement is None)


def settled_review(seeded):
    return next(i.review for i in seeded if i.settlement is not None)


def test_changes_only_the_fields_sent(client, seeded):
    review = unsettled_review(seeded)
    before = {
        "outcome": review.outcome,
        "corrected_type": review.corrected_type,
        "reviewer_id": review.reviewer_id,
        "created_at": review.created_at,
    }

    response = client.patch(url(review.id), json={"notes": "Checked again."})

    assert response.status_code == 200
    body = response.json()
    assert body["notes"] == "Checked again."
    assert body["outcome"] == before["outcome"]
    assert body["corrected_type"] == before["corrected_type"]
    assert body["reviewer_id"] == before["reviewer_id"]


def test_records_who_edited_and_when(client, seeded):
    review = unsettled_review(seeded)
    created_at = client.get(f"/api/v1/incidents/{review.incident_id}").json()["review"][
        "created_at"
    ]

    body = client.patch(url(review.id), json={"outcome": "escalated"}).json()

    assert body["outcome"] == "escalated"
    assert body["updated_by"] == 1
    assert body["updated_at"] is not None
    assert body["created_at"] == created_at


def test_null_clears_corrected_type(client, seeded):
    review = next(i.review for i in seeded if i.review and i.review.corrected_type)

    body = client.patch(url(review.id), json={"corrected_type": None}).json()

    assert body["corrected_type"] is None


def test_edit_shows_on_the_incident(client, seeded):
    review = unsettled_review(seeded)

    body = client.patch(url(review.id), json={"notes": "Updated"}).json()

    assert (
        client.get(f"/api/v1/incidents/{review.incident_id}").json()["review"] == body
    )


def test_settled_review_must_stay_escalated(client, seeded):
    review = settled_review(seeded)

    response = client.patch(url(review.id), json={"outcome": "dismissed"})

    assert response.status_code == 409


def test_settled_review_notes_can_still_change(client, seeded):
    review = settled_review(seeded)

    response = client.patch(url(review.id), json={"notes": "More detail."})

    assert response.status_code == 200
    assert response.json()["outcome"] == "escalated"


def test_missing_review_returns_404(client, seeded):
    missing_id = max(i.review.id for i in seeded if i.review) + 1000

    response = client.patch(url(missing_id), json={"notes": "x"})

    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"outcome": None},
        {"outcome": "approved"},
        {"severity": "minor"},
        {"corrected_type": ""},
    ],
)
def test_rejects_invalid_bodies(client, seeded, body):
    response = client.patch(url(unsettled_review(seeded).id), json=body)

    assert response.status_code == 422
