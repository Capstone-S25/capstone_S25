import hashlib
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from argon2 import PasswordHasher
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from data_pipeline.database import models  # noqa: F401
from data_pipeline.database.base import Base
from data_pipeline.database.models import User, UserSession

SESSION_COOKIE = "cctv_session"
PASSWORD = "Test-password-1"


@pytest.fixture
def postgres_session():
    """Isolate PostgreSQL tests in a transactional schema, leaving no rows."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL is required for database tests in CI")
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL database tests")
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                schema = "test_database_" + uuid4().hex
                connection.execute(
                    text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
                )
                connection.execute(text(f"CREATE SCHEMA {schema}"))
                # public is searched second only for the pgvector type.
                connection.execute(text(f"SET LOCAL search_path TO {schema}, public"))
                # checkfirst=False: tables already in public must not stop the
                # test copies being created in the test schema.
                Base.metadata.create_all(connection, checkfirst=False)
                with Session(
                    bind=connection, join_transaction_mode="create_savepoint"
                ) as db:
                    yield db
            finally:
                transaction.rollback()
    finally:
        engine.dispose()


@dataclass(frozen=True)
class Account:
    """A user with a live session token, as after signing in."""

    user_id: int
    email: str
    password: str
    token: str

    @property
    def cookies(self):
        return {SESSION_COOKIE: self.token}

    @property
    def bearer(self):
        return {"Authorization": "Bearer " + self.token}


@pytest.fixture
def make_account(postgres_session):
    """Create accounts like cayla-backend does: make_account(role="admin").

    The password (PASSWORD) is an argon2 hash and only the SHA-256 of the
    session token is stored, so the real session check accepts the token.
    """

    def make(*, role="staff", must_change_password=False, expires_in=None):
        user = User(
            name="Test " + role,
            email=uuid4().hex + "@example.com",
            role=role,
            password_hash=PasswordHasher().hash(PASSWORD),
            must_change_password=must_change_password,
        )
        postgres_session.add(user)
        postgres_session.flush()
        token = secrets.token_urlsafe(32)
        postgres_session.add(
            UserSession(
                token_hash=hashlib.sha256(token.encode()).digest(),
                user_id=user.user_id,
                expires_at=datetime.now(UTC) + (expires_in or timedelta(hours=1)),
            )
        )
        postgres_session.flush()
        return Account(user.user_id, user.email, PASSWORD, token)

    return make
