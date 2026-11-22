PUBLIC = {
    ("POST", "/api/v1/auth/login"),
    # "Forgot password?" is for people who can't sign in.
    ("POST", "/api/v1/auth/forgot-password"),
    ("POST", "/api/v1/auth/reset-password"),
}


def test_every_route_but_signing_in_needs_a_session(client, seeded):
    paths = client.get("/openapi.json").json()["paths"]
    routes = [
        (
            method.upper(),
            path.replace("{incident_id}", str(seeded[0].id))
            .replace("{review_id}", "1")
            .replace("{user_id}", "1"),
        )
        for path, operations in paths.items()
        for method in operations
    ]
    client.cookies.clear()

    # 9 from Person B plus the thumbnail; 11 login, user and audit-log routes.
    assert len(routes) == 21
    for method, path in routes:
        if (method, path) in PUBLIC:
            continue
        response = client.request(method, path, json={})
        assert response.status_code == 401, (method, path)
