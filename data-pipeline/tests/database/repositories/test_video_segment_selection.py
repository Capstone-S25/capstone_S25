from uuid import uuid4

from database.models.video_segments import SegmentStatus, VideoSegment
from database.repositories.video_segments import (
    get_oldest_video_segment,
    update_video_segment_status,
)
from video_ingestion.scheduling.worker import claim_next_segment


def test_selection_filters_source_status_and_orders_oldest_first(postgres_session):
    db = postgres_session
    session_id = uuid4()
    rows = [
        VideoSegment(
            source_id=source,
            session_id=session_id,
            path=f"{index}.mp4",
            start_at_ms=start,
            end_at_ms=start + 1000,
            status=status,
        )
        for index, (source, start, status) in enumerate(
            [
                ("other_camera", 0, SegmentStatus.PENDING),
                ("camera_01", 0, SegmentStatus.PROCESSED),
                ("camera_01", 2000, SegmentStatus.PENDING),
                ("camera_01", 1000, SegmentStatus.PENDING),
                ("camera_01", 1000, SegmentStatus.PENDING),
            ]
        )
    ]
    db.add_all(rows)
    db.flush()

    selected = get_oldest_video_segment(
        db, source_id="camera_01", status=SegmentStatus.PENDING
    )

    assert selected is not None
    assert selected is rows[3]
    assert selected.status is SegmentStatus.PENDING
    update_video_segment_status(db, selected, status=SegmentStatus.PROCESSING)
    db.refresh(selected)
    assert selected.status is SegmentStatus.PROCESSING
    assert (
        get_oldest_video_segment(
            db, source_id="camera_01", status=SegmentStatus.PENDING
        )
        is rows[4]
    )


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

    assert claim_next_segment(db, source_id="camera_01") is segment
    db.refresh(segment)
    assert segment.status is SegmentStatus.PROCESSING
    assert claim_next_segment(db, source_id="camera_01") is None

    # Rolling back the caller's transaction makes the row eligible again.
    db.rollback()
    db.refresh(segment)
    assert segment.status is SegmentStatus.PENDING
