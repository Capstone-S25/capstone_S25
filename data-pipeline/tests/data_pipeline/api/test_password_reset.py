"""Self-service "Forgot password?" (cayla-backend/app/routers/auth.py)."""

import sys
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import select, update

from data_pipeline.database.models import AuditLogEntry, PasswordReset

NEW_PASSWORD = "Brand-new-pass-3"


@pytest.fixture
def sent_emails(client, monkeypatch):
    """The emails the reset routes would send, as (to, Email) pairs."""
    routes = sys.modules["app.routers.auth"]  # cayla-backend/app, loaded by the API
    sent = []
    monkeypatch.setattr(
        routes, "send_email", lambda to, email: sent.append((to, email))
    )
    return sent


def reset_token(email):
    """The token in the emailed /reset-password?token=... link."""
    link = next(word for word in email.text.split() if "/reset-password?" in word)
    return parse_qs(urlsplit(link).query)["token"][0]


def sign_in(client, email, password):
    return client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )


def test_a_reset_link_sets_a_new_password_once_and_signs_out_everywhere(
    client, account, sent_emails, postgres_session
):
    response = client.post(
        "/api/v1/auth/forgot-password", json={"email": account.email.upper()}
    )

    assert response.status_code == 202
    [(to, email)] = sent_emails
    assert to == account.email
    token = reset_token(email)
    # Until the link is used, the current password keeps working.
    assert sign_in(client, account.email, account.password).status_code == 200

    response = client.post(
        "/api/v1/auth/reset-password",
        json={
            "token": token,
            "new_password": NEW_PASSWORD,
            "confirm_password": NEW_PASSWORD,
        },
    )

    assert response.status_code == 204
    client.cookies.clear()
    assert client.get("/api/v1/me", headers=account.bearer).status_code == 401
    assert sign_in(client, account.email, account.password).status_code == 401
    assert sign_in(client, account.email, NEW_PASSWORD).status_code == 200
    again = client.post(
        "/api/v1/auth/reset-password",
        json={"token": token, "new_password": "Another-pass-4"},
    )
    assert again.status_code == 400
    actions = set(
        postgres_session.scalars(
            select(AuditLogEntry.action).where(
                AuditLogEntry.entity_id == str(account.user_id)
            )
        )
    )
    assert {"request_password_reset", "reset_password_by_link"} <= actions


def test_an_unknown_email_gets_the_same_answer_and_no_email(
    client, account, sent_emails
):
    unknown = client.post(
        "/api/v1/auth/forgot-password", json={"email": "nobody@example.com"}
    )
    known = client.post("/api/v1/auth/forgot-password", json={"email": account.email})

    assert (unknown.status_code, unknown.json()) == (known.status_code, known.json())
    assert [to for to, _ in sent_emails] == [account.email]


def test_asking_again_within_a_minute_sends_no_second_email(
    client, account, sent_emails
):
    for _ in range(2):
        response = client.post(
            "/api/v1/auth/forgot-password", json={"email": account.email}
        )
        assert response.status_code == 202

    assert len(sent_emails) == 1


def test_a_weak_password_is_refused_and_the_link_still_works(
    client, account, sent_emails
):
    client.post("/api/v1/auth/forgot-password", json={"email": account.email})
    token = reset_token(sent_emails[0][1])

    weak = client.post(
        "/api/v1/auth/reset-password", json={"token": token, "new_password": "abc"}
    )
    good = client.post(
        "/api/v1/auth/reset-password",
        json={"token": token, "new_password": NEW_PASSWORD},
    )

    assert weak.status_code == 422
    assert good.status_code == 204


def test_an_expired_link_is_refused(client, account, sent_emails, postgres_session):
    client.post("/api/v1/auth/forgot-password", json={"email": account.email})
    token = reset_token(sent_emails[0][1])
    postgres_session.execute(
        update(PasswordReset)
        .where(PasswordReset.user_id == account.user_id)
        .values(expires_at=datetime.now(UTC) - timedelta(minutes=1))
    )
    postgres_session.flush()

    response = client.post(
        "/api/v1/auth/reset-password",
        json={"token": token, "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 400
    assert sign_in(client, account.email, account.password).status_code == 200
