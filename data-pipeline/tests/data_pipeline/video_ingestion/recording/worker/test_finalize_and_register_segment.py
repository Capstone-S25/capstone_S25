from unittest.mock import MagicMock, Mock

from sqlalchemy.exc import OperationalError

from data_pipeline.video_ingestion.recording import worker
from data_pipeline.video_ingestion.recording.segments import SegmentEntry


def test_finalizes_once_when_database_registration_retries(
    tmp_path, monkeypatch, make_source
):
    # A second finalization would fail because .part has already been renamed.
    part = tmp_path / "000000.mp4.part"
    part.write_bytes(b"recorded video")
    finalize = Mock(wraps=worker.finalize_segment_with_retry)
    insert = Mock(
        side_effect=[
            OperationalError(None, None, Exception("Connection lost")),
            True,
        ]
    )
    factory = MagicMock()
    monkeypatch.setattr(worker, "finalize_segment_with_retry", finalize)
    monkeypatch.setattr(worker, "insert_video_segment_metadata", insert)
    monkeypatch.setattr(worker.time, "sleep", Mock())

    segment = worker._finalize_and_register_segment(
        SegmentEntry("000000.mp4.part", 0, 1),
        tmp_path,
        source=make_source(),
        session_id="test-session",
        storage_root=tmp_path,
        session_started_at_utc=1_700_000_000_000,
        session_factory=factory,
        retention_hours=24,
    )

    finalize.assert_called_once()
    assert insert.call_count == 2
    assert factory.begin.call_count == 2
    assert not part.exists()
    assert segment.path == tmp_path / "000000.mp4"
    assert segment.path.read_bytes() == b"recorded video"
