from uuid import uuid4

from database.models.video_segments import SegmentStatus, VideoSegment
from database.repositories.video_segments import (
    get_oldest_video_segment,
    update_video_segment_status,
)


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
