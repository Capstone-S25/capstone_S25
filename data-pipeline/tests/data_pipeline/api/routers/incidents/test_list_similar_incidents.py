from datetime import timedelta

import numpy as np
import pytest

from data_pipeline.database.models import VLMAssessment, VLMResult


def url(incident_id):
    return f"/api/v1/incidents/{incident_id}/similar"


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def expected_similar(seeded, target, limit):
    scores = [
        (cosine(target.vlm_results[-1].embedding, i.vlm_results[-1].embedding), i.id)
        for i in seeded
        if i.id != target.id and i.vlm_results[-1].embedding is not None
    ]
    return [i for _, i in sorted(scores, key=lambda s: (-s[0], s[1]))[:limit]]


def test_most_similar_first(client, seeded):
    target = seeded[0]

    body = client.get(url(target.id)).json()

    assert [item["id"] for item in body] == expected_similar(seeded, target, 5)
    scores = [item["similarity"] for item in body]
    assert scores == sorted(scores, reverse=True)
    assert all(-1 <= score <= 1 for score in scores)


def test_matches_incidents_of_the_same_type(client, seeded):
    target = seeded[0]
    types = {i.id: i.event_type for i in seeded}

    body = client.get(url(target.id)).json()

    assert {types[item["id"]] for item in body} == {target.event_type}


def test_excludes_the_incident_itself(client, seeded):
    body = client.get(url(seeded[0].id), params={"limit": 20}).json()

    assert seeded[0].id not in [item["id"] for item in body]
    assert len(body) == 20


def test_items_carry_list_fields(client, seeded):
    item = client.get(url(seeded[0].id), params={"limit": 1}).json()[0]
    match = next(i for i in seeded if i.id == item["id"])

    assert item["event_type"] == match.event_type
    assert item["severity"] == match.severity
    assert item["review_outcome"] == (match.review.outcome if match.review else None)


def test_uses_the_newest_vlm_result(client, seeded, postgres_session):
    target = seeded[0]
    target.vlm_results.append(
        VLMResult(
            event_type=target.event_type,
            assessment=VLMAssessment.CONFIRMED,
            model_version="rerun",
            processed_at=target.vlm_results[0].processed_at + timedelta(hours=1),
            embedding=None,
        )
    )
    postgres_session.flush()

    assert client.get(url(target.id)).json() == []


def test_skips_incidents_without_an_embedding(client, seeded, postgres_session):
    target, other = seeded[0], seeded[1]
    other.vlm_results[0].embedding = None
    postgres_session.flush()

    body = client.get(url(target.id), params={"limit": 20}).json()

    ids = [item["id"] for item in body]
    assert other.id not in ids
    assert ids == expected_similar(seeded, target, 20)


@pytest.mark.parametrize("limit", [0, 21])
def test_rejects_limit_out_of_range(client, seeded, limit):
    response = client.get(url(seeded[0].id), params={"limit": limit})

    assert response.status_code == 422


def test_missing_incident_returns_404(client, seeded):
    response = client.get(url(max(i.id for i in seeded) + 1000))

    assert response.status_code == 404
