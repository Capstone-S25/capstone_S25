import pytest
from fastapi import HTTPException

from data_pipeline.api.auth import get_current_user


def reject_login():
    raise HTTPException(401, "Not logged in")


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/incidents",
        "/api/v1/incidents/{id}",
        "/api/v1/incidents/{id}/detections",
    ],
)
def test_every_route_uses_the_login_dependency(client, seeded, path):
    client.app.dependency_overrides[get_current_user] = reject_login

    response = client.get(path.format(id=seeded[0].id))

    assert response.status_code == 401
