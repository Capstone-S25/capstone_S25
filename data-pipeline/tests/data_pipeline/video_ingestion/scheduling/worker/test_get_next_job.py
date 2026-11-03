from contextlib import contextmanager
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from data_pipeline.database.models.video_segments import SegmentStatus, VideoSegment
from data_pipeline.video_ingestion.scheduling import worker


@pytest.fixture
def scheduling_context(monkeypatch):
    events = []
    factory = Mock()

    @contextmanager
    def transaction():
        db = Mock()
        events.append("begin")
        try:
            yield db
        except BaseException:
            events.append("rollback")
            raise
        else:
            events.append("commit")

    factory.begin.side_effect = transaction
    claim = Mock()
    monkeypatch.setattr(worker, "_claim_next_segment", claim)
    return factory, claim, events


def segment(source_id: str) -> VideoSegment:
    """Build an unsaved segment without opening a database connection."""
    return VideoSegment(
        id=1,
        source_id=source_id,
        session_id=uuid4(),
        path=f"{source_id}/session/000000.mp4",
        start_at_ms=1000,
        end_at_ms=6000,
    )


def checked_cameras(claim):
    return [call.kwargs["source_id"] for call in claim.call_args_list]


def test_returns_metadata_after_commit(scheduling_context):
    factory, claim, events = scheduling_context
    row = segment("camera_01")
    claim.return_value = row
    scheduler = worker.SegmentScheduler(
        source_ids=["camera_01", "camera_02"], session_factory=factory
    )

    job = scheduler.get_next_job()

    assert job == worker.SegmentJob(
        segment_id=row.id,
        source_id=row.source_id,
        session_id=row.session_id,
        path=row.path,
        start_at_ms=row.start_at_ms,
        end_at_ms=row.end_at_ms,
    )
    assert events == ["begin", "commit"]
    assert checked_cameras(claim) == ["camera_01"]


def test_skips_empty_camera_and_keeps_cursor_for_next_call(scheduling_context):
    factory, claim, events = scheduling_context
    claim.side_effect = [None, segment("camera_02"), segment("camera_03")]
    scheduler = worker.SegmentScheduler(
        source_ids=["camera_01", "camera_02", "camera_03"], session_factory=factory
    )

    first = scheduler.get_next_job()
    second = scheduler.get_next_job()

    assert first is not None
    assert second is not None
    assert first.source_id == "camera_02"
    assert second.source_id == "camera_03"
    assert checked_cameras(claim) == ["camera_01", "camera_02", "camera_03"]
    assert events == ["begin", "commit"] * 3


def test_cursor_wraps_and_checks_every_camera_once_when_idle(scheduling_context):
    factory, claim, events = scheduling_context
    claim.side_effect = [segment("camera_01"), None, None, None]
    scheduler = worker.SegmentScheduler(
        source_ids=["camera_01", "camera_02", "camera_03"], session_factory=factory
    )

    job = scheduler.get_next_job()
    assert job is not None
    assert job.source_id == "camera_01"
    assert scheduler.get_next_job() is None

    assert checked_cameras(claim) == [
        "camera_01",
        "camera_02",
        "camera_03",
        "camera_01",
    ]
    assert events == ["begin", "commit"] * 4


def test_empty_camera_list_does_not_open_transaction(scheduling_context):
    factory, claim, events = scheduling_context
    scheduler = worker.SegmentScheduler(source_ids=[], session_factory=factory)

    assert scheduler.get_next_job() is None

    assert not events
    claim.assert_not_called()


def test_claim_error_rolls_back_and_advances_cursor(scheduling_context):
    factory, claim, events = scheduling_context
    claim.side_effect = [RuntimeError("claim failed"), segment("camera_02")]
    scheduler = worker.SegmentScheduler(
        source_ids=["camera_01", "camera_02"], session_factory=factory
    )

    with pytest.raises(RuntimeError, match="claim failed"):
        scheduler.get_next_job()
    job = scheduler.get_next_job()
    assert job is not None
    assert job.source_id == "camera_02"

    assert checked_cameras(claim) == ["camera_01", "camera_02"]
    assert events == ["begin", "rollback", "begin", "commit"]


def test_deduplicates_camera_ids_in_first_occurrence_order(scheduling_context):
    factory, claim, events = scheduling_context
    claim.return_value = None
    scheduler = worker.SegmentScheduler(
        source_ids=["camera_02", "camera_01", "camera_02", "camera_03", "camera_01"],
        session_factory=factory,
    )

    assert scheduler.get_next_job() is None

    assert checked_cameras(claim) == ["camera_02", "camera_01", "camera_03"]
    assert events == ["begin", "commit"] * 3


def test_round_robin_jobs_survive_session_close_and_claim_only_one_row(
    postgres_session,
):
    db = postgres_session
    recording_id = uuid4()
    rows = [
        VideoSegment(
            source_id=source_id,
            session_id=recording_id,
            path=f"{source_id}/{recording_id.hex}/{index:06d}.mp4",
            start_at_ms=index * 5000,
            end_at_ms=(index + 1) * 5000,
        )
        for source_id, index in [("camera_01", 0), ("camera_01", 1), ("camera_02", 0)]
    ]
    db.add_all(rows)
    db.flush()
    row_ids = [row.id for row in rows]
    db.commit()
    factory = sessionmaker(bind=db.get_bind(), join_transaction_mode="create_savepoint")
    scheduler = worker.SegmentScheduler(
        source_ids=["empty_camera", "camera_01", "camera_02"],
        session_factory=factory,
    )

    first = scheduler.get_next_job()

    assert isinstance(first, worker.SegmentJob)
    assert first.segment_id == row_ids[0]
    assert first.session_id == recording_id
    assert first.path == f"camera_01/{recording_id.hex}/000000.mp4"
    assert first.start_at_ms == 0
    assert first.end_at_ms == 5000
    statuses = dict(db.execute(select(VideoSegment.id, VideoSegment.status)).all())
    assert statuses == {
        row_ids[0]: SegmentStatus.PROCESSING,
        row_ids[1]: SegmentStatus.PENDING,
        row_ids[2]: SegmentStatus.PENDING,
    }
    # Finish the inspecting session's savepoint before opening another worker session.
    db.commit()

    second = scheduler.get_next_job()
    assert second is not None
    assert second.segment_id == row_ids[2]

    third = scheduler.get_next_job()
    assert third is not None
    assert third.segment_id == row_ids[1]
    assert scheduler.get_next_job() is None
