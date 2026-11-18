CLIP_BYTES = b"fake mp4 bytes for testing"


def url(incident_id):
    return f"/api/v1/incidents/{incident_id}/clip"


def write_clip(recordings_dir, incident):
    path = recordings_dir / incident.evidence_clip_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(CLIP_BYTES)


def test_streams_the_clip(client, seeded, recordings_dir):
    write_clip(recordings_dir, seeded[0])

    response = client.get(url(seeded[0].id))

    assert response.status_code == 200
    assert response.content == CLIP_BYTES
    assert response.headers["content-type"] == "video/mp4"


def test_supports_range_requests_for_seeking(client, seeded, recordings_dir):
    write_clip(recordings_dir, seeded[0])

    response = client.get(url(seeded[0].id), headers={"Range": "bytes=0-3"})

    assert response.status_code == 206
    assert response.content == CLIP_BYTES[:4]


def test_missing_file_returns_404(client, seeded):
    response = client.get(url(seeded[0].id))

    assert response.status_code == 404
    assert response.json() == {"detail": "Evidence file not found"}


def test_path_outside_recordings_folder_returns_404(
    client, seeded, recordings_dir, postgres_session
):
    (recordings_dir.parent / "secret.mp4").write_bytes(b"secret")
    seeded[0].evidence_clip_path = "../secret.mp4"
    postgres_session.flush()

    response = client.get(url(seeded[0].id))

    assert response.status_code == 404
    assert response.json() == {"detail": "Evidence file not found"}


def test_missing_incident_returns_404(client, seeded):
    response = client.get(url(max(i.id for i in seeded) + 1000))

    assert response.status_code == 404
    assert response.json() == {"detail": "Incident not found"}
