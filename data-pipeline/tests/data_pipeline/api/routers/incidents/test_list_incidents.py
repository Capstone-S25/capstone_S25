from datetime import timedelta

import pytest

URL = "/api/v1/incidents"


def get_ids(client, **params):
    response = client.get(URL, params={"page_size": 100, **params})
    assert response.status_code == 200, response.text
    body = response.json()
    ids = [item["id"] for item in body["items"]]
    assert body["total"] == len(ids)
    return ids


def outcome_of(incident):
    return incident.review.outcome if incident.review else "unreviewed"


def test_lists_all_incidents_newest_first(client, seeded):
    response = client.get(URL, params={"page_size": 100})

    body = response.json()
    assert body["total"] == 30
    assert [item["id"] for item in body["items"]] == [i.id for i in seeded]


def test_list_item_matches_incident(client, seeded):
    reviewed = next(i for i in seeded if i.review)

    item = next(
        item
        for item in client.get(URL, params={"page_size": 100}).json()["items"]
        if item["id"] == reviewed.id
    )

    assert item["camera_id"] == reviewed.camera_id
    assert item["location_id"] == reviewed.location_id
    assert item["event_type"] == reviewed.event_type
    assert item["severity"] == reviewed.severity
    assert item["review_outcome"] == reviewed.review.outcome


def test_unreviewed_item_has_no_outcome(client, seeded):
    unreviewed = next(i for i in seeded if i.review is None)

    items = client.get(URL, params={"outcome": "unreviewed", "page_size": 100})

    item = next(i for i in items.json()["items"] if i["id"] == unreviewed.id)
    assert item["review_outcome"] is None


@pytest.mark.parametrize(
    ("params", "matches"),
    [
        ({"location_id": 1}, lambda i: i.location_id == 1),
        ({"location_id": 3}, lambda i: i.location_id == 3),
        ({"camera_id": "camera_03"}, lambda i: i.camera_id == "camera_03"),
        ({"event_type": "loitering"}, lambda i: i.event_type == "loitering"),
        ({"severity": "minor"}, lambda i: i.severity == "minor"),
        ({"severity": "moderate"}, lambda i: i.severity == "moderate"),
        ({"severity": "critical"}, lambda i: i.severity == "critical"),
        ({"outcome": "accepted"}, lambda i: outcome_of(i) == "accepted"),
        ({"outcome": "dismissed"}, lambda i: outcome_of(i) == "dismissed"),
        ({"outcome": "escalated"}, lambda i: outcome_of(i) == "escalated"),
        ({"outcome": "unreviewed"}, lambda i: outcome_of(i) == "unreviewed"),
        (
            {"location_id": 1, "event_type": "littering", "severity": "minor"},
            lambda i: (
                (i.location_id, i.event_type, i.severity) == (1, "littering", "minor")
            ),
        ),
        (
            {"camera_id": "camera_01", "outcome": "unreviewed"},
            lambda i: i.camera_id == "camera_01" and i.review is None,
        ),
    ],
)
def test_filters(client, seeded, params, matches):
    expected = [i.id for i in seeded if matches(i)]

    assert expected, "seed data should include matches for this filter"
    assert get_ids(client, **params) == expected


def test_date_from_includes_the_whole_first_day(client, seeded, now):
    day = (now - timedelta(days=7)).date()
    expected = [i.id for i in seeded if i.first_anomaly_at.date() >= day]

    assert 0 < len(expected) < 30
    assert get_ids(client, date_from=day.isoformat()) == expected


def test_date_to_includes_the_whole_last_day(client, seeded, now):
    day = (now - timedelta(days=7)).date()
    expected = [i.id for i in seeded if i.first_anomaly_at.date() <= day]

    assert 0 < len(expected) < 30
    assert get_ids(client, date_to=day.isoformat()) == expected


def test_single_day_range(client, seeded):
    day = seeded[0].first_anomaly_at.date()
    expected = [i.id for i in seeded if i.first_anomaly_at.date() == day]

    assert get_ids(client, date_from=day.isoformat(), date_to=day.isoformat()) == (
        expected
    )


def test_date_range_combined_with_another_filter(client, seeded, now):
    start, end = (now - timedelta(days=10)).date(), (now - timedelta(days=3)).date()
    expected = [
        i.id
        for i in seeded
        if start <= i.first_anomaly_at.date() <= end and i.severity == "critical"
    ]

    assert expected
    assert (
        get_ids(
            client,
            date_from=start.isoformat(),
            date_to=end.isoformat(),
            severity="critical",
        )
        == expected
    )


def test_no_matches_returns_empty_page(client):
    body = client.get(URL, params={"camera_id": "camera_99"}).json()

    assert body == {"items": [], "total": 0, "page": 1, "page_size": 20}


def test_default_page_is_first_twenty(client, seeded):
    body = client.get(URL).json()

    assert body["page"] == 1
    assert body["page_size"] == 20
    assert body["total"] == 30
    assert [item["id"] for item in body["items"]] == [i.id for i in seeded[:20]]


def test_pages_continue_without_overlap(client, seeded):
    first = client.get(URL, params={"page_size": 12, "page": 1}).json()
    second = client.get(URL, params={"page_size": 12, "page": 2}).json()
    third = client.get(URL, params={"page_size": 12, "page": 3}).json()

    ids = [item["id"] for page in (first, second, third) for item in page["items"]]
    assert [len(page["items"]) for page in (first, second, third)] == [12, 12, 6]
    assert ids == [i.id for i in seeded]
    assert {page["total"] for page in (first, second, third)} == {30}


def test_page_past_the_end_is_empty_but_keeps_total(client):
    body = client.get(URL, params={"page": 5}).json()

    assert body["items"] == []
    assert body["total"] == 30


def test_filtered_total_counts_all_pages(client, seeded):
    expected = [i.id for i in seeded if i.review is None]

    body = client.get(URL, params={"outcome": "unreviewed", "page_size": 4}).json()

    assert body["total"] == len(expected)
    assert [item["id"] for item in body["items"]] == expected[:4]


def test_page_size_100_is_allowed(client):
    assert client.get(URL, params={"page_size": 100}).status_code == 200


@pytest.mark.parametrize(
    "params",
    [
        {"page": 0},
        {"page_size": 0},
        {"page_size": 101},
        {"severity": "high"},
        {"outcome": "pending"},
        {"date_from": "yesterday"},
        {"location_id": "main-gate"},
        {"date_from": "2026-10-20", "date_to": "2026-10-19"},
    ],
)
def test_rejects_invalid_parameters(client, params):
    assert client.get(URL, params=params).status_code == 422
