from unittest.mock import MagicMock, Mock

from sqlalchemy.exc import OperationalError

from video_ingestion import recording_worker


def test_finalizes_once_when_database_registration_retries(
    tmp_path, monkeypatch, make_source
):
    # A second finalization would fail because .part has already been renamed.
    part = tmp_path / "000000.mp4.part"
    part.write_bytes(b"recorded video")
    finalize = Mock(wraps=recording_worker._finalize_segment_with_retry)
    insert = Mock(
        side_effect=[
            OperationalError(None, None, Exception("Connection lost")),
            True,
        ]
    )
    factory = MagicMock()
    monkeypatch.setattr(recording_worker, "_finalize_segment_with_retry", finalize)
    monkeypatch.setattr(recording_worker, "insert_video_segment_metadata", insert)
    monkeypatch.setattr(recording_worker.time, "sleep", Mock())

    segment = recording_worker._finalize_and_register_segment(
        recording_worker.SegmentEntry("000000.mp4.part", 0, 1),
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
