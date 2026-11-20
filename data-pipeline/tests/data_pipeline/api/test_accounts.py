from sqlalchemy import select

from data_pipeline.database.models import AuditLogEntry


def test_signing_in_opens_a_session_for_the_incidents_api(client, make_account):
    staff = make_account()
    client.cookies.clear()

    response = client.post(
        "/api/v1/auth/login",
        json={"email": staff.email.upper(), "password": staff.password},
    )

    assert response.status_code == 200
    assert "cctv_session=" in response.headers["set-cookie"]
    body = response.json()
    assert body["user"]["user_id"] == staff.user_id
    bearer = {"Authorization": "Bearer " + body["token"]}
    assert client.get("/api/v1/me", headers=bearer).json()["user_id"] == staff.user_id
    assert client.get("/api/v1/incidents", headers=bearer).status_code == 200

    assert client.post("/api/v1/auth/logout", headers=bearer).status_code == 204
    assert client.get("/api/v1/incidents", headers=bearer).status_code == 401


def test_a_wrong_password_is_refused(client, account):
    response = client.post(
        "/api/v1/auth/login", json={"email": account.email, "password": "wrong"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_only_admins_list_the_users(client, account, make_account):
    admin = make_account(role="admin")

    assert client.get("/api/v1/users").status_code == 403
    response = client.get("/api/v1/users", headers=admin.bearer)
    assert response.status_code == 200
    assert {account.user_id, admin.user_id} <= {
        user["user_id"] for user in response.json()
    }


def test_choosing_a_password_unlocks_the_api_and_is_audited(
    client, make_account, postgres_session
):
    invited = make_account(must_change_password=True)

    response = client.post(
        "/api/v1/auth/change-password",
        headers=invited.bearer,
        json={"new_password": "New-password-2", "confirm_password": "New-password-2"},
    )

    assert response.status_code == 200
    assert client.get("/api/v1/incidents", headers=invited.bearer).status_code == 200
    entry = postgres_session.scalars(
        select(AuditLogEntry).where(AuditLogEntry.entity_id == str(invited.user_id))
    ).one()
    assert (entry.action, entry.actor_user_id) == ("change_password", invited.user_id)
