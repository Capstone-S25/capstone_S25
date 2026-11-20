from datetime import timedelta

from sqlalchemy import func, update

from data_pipeline.api.accounts import deps
from data_pipeline.api.auth import CurrentUser, Role, get_current_user
from data_pipeline.database.models import User, UserSession

INCIDENTS = "/api/v1/incidents"


def test_a_bearer_token_works_like_the_cookie(client, account):
    client.cookies.clear()

    assert client.get(INCIDENTS, headers=account.bearer).status_code == 200


def test_an_unknown_token_is_rejected(client):
    response = client.get(INCIDENTS, headers={"Authorization": "Bearer not-a-token"})

    assert response.status_code == 401


def test_an_expired_session_is_rejected(client, make_account):
    expired = make_account(expires_in=timedelta(hours=-1))

    assert client.get(INCIDENTS, headers=expired.bearer).status_code == 401


def test_a_signed_out_session_is_rejected(client, account, postgres_session):
    postgres_session.execute(
        update(UserSession)
        .where(UserSession.user_id == account.user_id)
        .values(revoked_at=func.now())
    )

    assert client.get(INCIDENTS).status_code == 401


def test_a_deactivated_user_is_rejected(client, account, postgres_session):
    postgres_session.execute(
        update(User).where(User.user_id == account.user_id).values(is_active=False)
    )

    assert client.get(INCIDENTS).status_code == 401


def test_a_temporary_password_must_be_changed_first(client, make_account):
    invited = make_account(must_change_password=True)

    response = client.get(INCIDENTS, headers=invited.bearer)

    assert response.status_code == 403
    assert response.json()["detail"] == "Password change required"


def test_the_user_comes_from_the_account():
    user = deps.User(user_id=7, name="Ada", email="ada@example.com", role="admin")

    assert get_current_user(user) == CurrentUser(id=7, name="Ada", role=Role.ADMIN)
