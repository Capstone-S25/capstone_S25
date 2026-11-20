def url(incident_id):
    return f"/api/v1/incidents/{incident_id}/rerun-vlm"


def test_queues_a_rerun(client, seeded, account):
    response = client.post(url(seeded[0].id))

    assert response.status_code == 202
    body = response.json()
    assert body["incident_id"] == seeded[0].id
    assert body["requested_by"] == account.user_id
    assert body["requested_at"] is not None


def test_incident_detail_shows_the_waiting_request(client, seeded):
    detail = f"/api/v1/incidents/{seeded[0].id}"
    assert client.get(detail).json()["vlm_rerun_requested_at"] is None

    requested_at = client.post(url(seeded[0].id)).json()["requested_at"]

    assert client.get(detail).json()["vlm_rerun_requested_at"] == requested_at


def test_asking_twice_keeps_the_first_request(client, seeded):
    first = client.post(url(seeded[0].id)).json()

    second = client.post(url(seeded[0].id))

    assert second.status_code == 202
    assert second.json() == first


def test_missing_incident_returns_404(client, seeded):
    response = client.post(url(max(i.id for i in seeded) + 1000))

    assert response.status_code == 404
