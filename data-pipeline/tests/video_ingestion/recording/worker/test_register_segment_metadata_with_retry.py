from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError

from video_ingestion.recording import worker
from video_ingestion.recording.segments import FinalizedSegment


@pytest.fixture
def registration(tmp_path, monkeypatch):
    """Supply existing segment metadata; no file or finalization is needed."""
    factory = MagicMock()
    insert = Mock(return_value=True)
    sleep = Mock()
    monkeypatch.setattr(worker, "insert_video_segment_metadata", insert)
    monkeypatch.setattr(worker.time, "sleep", sleep)
    segment = FinalizedSegment(
        path=tmp_path / "000000.mp4",
        start_at_ms=1_700_000_000_000,
        end_at_ms=1_700_000_001_000,
    )
    return factory, insert, sleep, segment


def _run_registration(registration, tmp_path, **kwargs):
    factory, _, _, segment = registration
    return worker._register_segment_metadata_with_retry(
        segment,
        source_id="camera_01",
        session_id="test-session",
        storage_root=tmp_path,
        session_factory=factory,
        retention_hours=24,
        **kwargs,
    )


def test_registers_metadata_without_retry(registration, tmp_path):
    factory, insert, sleep, segment = registration

    _run_registration(registration, tmp_path)

    insert.assert_called_once()
    metadata = insert.call_args.kwargs
    assert metadata["source_id"] == "camera_01"
    assert metadata["session_id"] == "test-session"
    assert metadata["storage_root"] == tmp_path
    assert metadata["path"] == segment.path
    assert metadata["start_at_ms"] == segment.start_at_ms
    assert metadata["end_at_ms"] == segment.end_at_ms
    factory.begin.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("error_type", [OperationalError, InterfaceError])
def test_retries_insert_with_unchanged_metadata(registration, tmp_path, error_type):
    factory, insert, sleep, _ = registration
    insert.side_effect = [error_type(None, None, Exception("Connection lost")), True]

    _run_registration(registration, tmp_path)

    assert factory.begin.call_count == 2
    assert insert.call_count == 2
    # Retry exactly the same metadata, including retention deadline.
    assert insert.call_args_list[0] == insert.call_args_list[1]
    sleep.assert_called_once_with(0.25)


@pytest.mark.parametrize("max_attempts", [1, 3])
def test_reraises_after_retry_limit(registration, tmp_path, max_attempts):
    factory, insert, sleep, _ = registration
    error = OperationalError(None, None, Exception("Database unavailable"))
    insert.side_effect = error

    with pytest.raises(OperationalError) as raised:
        _run_registration(registration, tmp_path, max_attempts=max_attempts)

    assert raised.value is error
    assert factory.begin.call_count == max_attempts
    assert insert.call_count == max_attempts
    assert sleep.call_count == max_attempts - 1


def test_integrity_error_is_not_retried(registration, tmp_path):
    factory, insert, sleep, _ = registration
    insert.side_effect = IntegrityError(None, None, Exception("Constraint violation"))

    with pytest.raises(IntegrityError):
        _run_registration(registration, tmp_path)

    factory.begin.assert_called_once()
    insert.assert_called_once()
    sleep.assert_not_called()
