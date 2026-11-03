from unittest.mock import Mock
from uuid import uuid4

from sqlalchemy.orm import Session

from data_pipeline.database.models.video_segments import SegmentStatus, VideoSegment
from data_pipeline.video_ingestion.scheduling import worker


def test_claims_pending_segment_with_lock_and_processing_status(monkeypatch):
    db = Mock(spec=Session)
    segment = Mock(spec=VideoSegment)
    select_segment = Mock(return_value=segment)
    update_status = Mock()
    monkeypatch.setattr(worker, "get_oldest_video_segment", select_segment)
    monkeypatch.setattr(worker, "update_video_segment_status", update_status)

    result = worker._claim_next_segment(db, source_id="camera_01")

    assert result is segment
    select_segment.assert_called_once_with(
        db, source_id="camera_01", status=SegmentStatus.PENDING, for_update=True
    )
    update_status.assert_called_once_with(db, segment, status=SegmentStatus.PROCESSING)
    # The caller commits after claiming, before handing the segment to decoding.
    db.commit.assert_not_called()


def test_no_available_segment_does_not_update_status(monkeypatch):
    db = Mock(spec=Session)
    monkeypatch.setattr(worker, "get_oldest_video_segment", Mock(return_value=None))
    update_status = Mock()
    monkeypatch.setattr(worker, "update_video_segment_status", update_status)

    assert worker._claim_next_segment(db, source_id="camera_01") is None

    update_status.assert_not_called()
    db.commit.assert_not_called()


def test_claim_updates_status_without_claiming_same_row_twice(postgres_session):
    db = postgres_session
    segment = VideoSegment(
        source_id="camera_01",
        session_id=uuid4(),
        path="clip.mp4",
        start_at_ms=0,
        end_at_ms=1000,
    )
    db.add(segment)
    db.commit()

    assert worker._claim_next_segment(db, source_id="camera_01") is segment
    db.refresh(segment)
    assert segment.status is SegmentStatus.PROCESSING
    assert worker._claim_next_segment(db, source_id="camera_01") is None

    # Rolling back the caller's transaction makes the row eligible again.
    db.rollback()
    db.refresh(segment)
    assert segment.status is SegmentStatus.PENDING
