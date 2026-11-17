from fastapi import HTTPException

from data_pipeline.api.auth import get_current_user


def reject_login():
    raise HTTPException(401, "Not logged in")


def test_every_route_uses_the_login_dependency(client, seeded):
    paths = client.get("/openapi.json").json()["paths"]
    routes = [
        (
            method.upper(),
            path.replace("{incident_id}", str(seeded[0].id)).replace(
                "{review_id}", "1"
            ),
        )
        for path, operations in paths.items()
        for method in operations
    ]
    client.app.dependency_overrides[get_current_user] = reject_login

    assert len(routes) == 9
    for method, path in routes:
        response = client.request(method, path, json={})
        assert response.status_code == 401, (method, path)
