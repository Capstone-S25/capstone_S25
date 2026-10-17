from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models.video_segments import SegmentStatus, VideoSegment
from database.repositories.video_segments import insert_video_segment_metadata


@pytest.fixture
def segment_data(tmp_path):
    """Valid insertion arguments for one finalized segment."""
    session_id = uuid4().hex
    storage_root = tmp_path / "recordings"
    return {
        "source_id": "camera_01",
        "session_id": session_id,
        "storage_root": storage_root,
        "path": storage_root / "camera_01" / session_id / "000000.mp4",
        "start_at_ms": 1_700_000_000_000,
        "end_at_ms": 1_700_000_005_000,
    }


def test_rejects_path_outside_storage_before_query(segment_data, tmp_path):
    db = Mock(spec=Session)
    segment_data["path"] = tmp_path / "outside.mp4"

    with pytest.raises(ValueError):
        insert_video_segment_metadata(db, **segment_data)

    db.execute.assert_not_called()


def test_rejects_invalid_session_uuid_before_query(segment_data):
    db = Mock(spec=Session)
    segment_data["session_id"] = "invalid"

    with pytest.raises(ValueError):
        insert_video_segment_metadata(db, **segment_data)

    db.execute.assert_not_called()


@pytest.mark.parametrize("delete_at", [None, datetime(2030, 1, 1, tzinfo=UTC)])
def test_inserts_normalized_metadata_and_leaves_transaction_open(
    postgres_session, segment_data, delete_at
):
    db = postgres_session
    expected_path = f"camera_01/{segment_data['session_id']}/000000.mp4"

    inserted = insert_video_segment_metadata(db, **segment_data, delete_at=delete_at)

    assert inserted is True
    assert db.in_transaction()

    stored_segment = db.scalar(select(VideoSegment))
    assert stored_segment is not None
    assert stored_segment.source_id == segment_data["source_id"]
    assert stored_segment.session_id == UUID(segment_data["session_id"])
    assert stored_segment.path == expected_path
    assert stored_segment.start_at_ms == segment_data["start_at_ms"]
    assert stored_segment.end_at_ms == segment_data["end_at_ms"]
    assert stored_segment.created_at is not None
    assert stored_segment.created_at.utcoffset() is not None
    assert stored_segment.status is SegmentStatus.PENDING
    assert stored_segment.delete_at == delete_at

    # The caller can still undo the insertion: the function did not commit.
    db.rollback()
    remaining_segments = db.scalar(select(func.count()).select_from(VideoSegment))
    assert remaining_segments == 0


def test_duplicate_path_is_noop_and_session_remains_usable(
    postgres_session, segment_data
):
    db = postgres_session
    inserted = insert_video_segment_metadata(db, **segment_data)
    assert inserted is True

    # Use a different spelling of the same path and attempt to change its metadata.
    original_path = segment_data["path"]
    duplicate_data = segment_data.copy()
    duplicate_data["path"] = original_path.parent / "nested" / ".." / original_path.name
    duplicate_data["end_at_ms"] = 1_700_000_009_000

    duplicate_inserted = insert_video_segment_metadata(db, **duplicate_data)

    assert duplicate_inserted is False
    stored_end_at_ms = db.scalar(select(VideoSegment.end_at_ms))
    assert stored_end_at_ms == segment_data["end_at_ms"]

    # A conflict must not prevent another insertion in the same transaction.
    next_segment_data = segment_data.copy()
    next_segment_data["path"] = original_path.with_name("000001.mp4")

    next_inserted = insert_video_segment_metadata(db, **next_segment_data)

    assert next_inserted is True
    db.commit()
    segment_count = db.scalar(select(func.count()).select_from(VideoSegment))
    assert segment_count == 2


def test_rejects_naive_delete_at_before_query(segment_data):
    db = Mock(spec=Session)

    with pytest.raises(ValueError, match="timezone-aware"):
        insert_video_segment_metadata(
            db, **segment_data, delete_at=datetime(2026, 10, 17)
        )

    db.execute.assert_not_called()


def test_status_update_preserves_created_at(postgres_session, segment_data):
    db = postgres_session
    insert_video_segment_metadata(db, **segment_data)
    segment = db.scalar(select(VideoSegment))
    # Use an older registration time so an accidental timestamp reset is visible,
    # even when the database's now() is constant within this transaction.
    segment.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    db.flush()
    created_at = segment.created_at

    segment.status = SegmentStatus.PROCESSED
    db.flush()
    db.refresh(segment)

    assert segment.status is SegmentStatus.PROCESSED
    assert segment.created_at == created_at


def test_duplicate_insert_preserves_lifecycle_metadata(postgres_session, segment_data):
    db = postgres_session
    deadline = datetime(2030, 1, 1, tzinfo=UTC)
    insert_video_segment_metadata(db, **segment_data, delete_at=deadline)
    segment = db.scalar(select(VideoSegment))
    segment.status = SegmentStatus.PROCESSED
    segment.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    db.flush()
    created_at = segment.created_at

    # Retrying registration must not reset status, creation time, or retention.
    assert insert_video_segment_metadata(db, **segment_data) is False
    db.refresh(segment)
    assert segment.status is SegmentStatus.PROCESSED
    assert segment.created_at == created_at
    assert segment.delete_at == deadline
