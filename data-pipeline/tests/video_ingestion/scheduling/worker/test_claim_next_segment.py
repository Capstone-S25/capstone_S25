from unittest.mock import Mock

from sqlalchemy.orm import Session

from database.models.video_segments import SegmentStatus, VideoSegment
from video_ingestion.scheduling import worker


def test_claims_pending_segment_with_lock_and_processing_status(monkeypatch):
    db = Mock(spec=Session)
    segment = Mock(spec=VideoSegment)
    select_segment = Mock(return_value=segment)
    update_status = Mock()
    monkeypatch.setattr(worker, "get_oldest_video_segment", select_segment)
    monkeypatch.setattr(worker, "update_video_segment_status", update_status)

    result = worker.claim_next_segment(db, source_id="camera_01")

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

    assert worker.claim_next_segment(db, source_id="camera_01") is None

    update_status.assert_not_called()
    db.commit.assert_not_called()
