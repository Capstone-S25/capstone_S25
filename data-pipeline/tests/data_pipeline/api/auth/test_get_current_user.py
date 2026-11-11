import pytest

from data_pipeline.api.auth import Role, get_current_user


def test_defaults_to_admin(monkeypatch):
    monkeypatch.delenv("DEV_USER_ROLE", raising=False)

    assert get_current_user().role == Role.ADMIN


def test_role_comes_from_environment(monkeypatch):
    monkeypatch.setenv("DEV_USER_ROLE", "member")

    assert get_current_user().role == Role.MEMBER


def test_rejects_unknown_role(monkeypatch):
    monkeypatch.setenv("DEV_USER_ROLE", "superuser")

    with pytest.raises(RuntimeError, match="DEV_USER_ROLE"):
        get_current_user()
