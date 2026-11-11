import os
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from data_pipeline.api.app import create_app
from data_pipeline.api.database import get_db
from scripts.seed_test_data import seed_test_data


@pytest.fixture
def now():
    """Fixed seed time so date filters are tested on known days."""
    return datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def seeded(postgres_session, now):
    """Seed incidents into the test schema; returns them newest first."""
    incidents = seed_test_data(postgres_session, now=now)
    return sorted(incidents, key=lambda i: (i.first_anomaly_at, i.id), reverse=True)


@pytest.fixture
def client(postgres_session, seeded):
    """API client whose requests use the seeded test session."""
    app = create_app(os.environ["TEST_DATABASE_URL"])

    def override_get_db():
        yield postgres_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
