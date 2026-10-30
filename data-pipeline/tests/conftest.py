import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from data_pipeline.database import models  # noqa: F401
from data_pipeline.database.base import Base


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
                connection.execute(text(f"CREATE SCHEMA {schema}"))
                connection.execute(text(f"SET LOCAL search_path TO {schema}"))
                Base.metadata.create_all(connection)
                with Session(
                    bind=connection, join_transaction_mode="create_savepoint"
                ) as db:
                    yield db
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
