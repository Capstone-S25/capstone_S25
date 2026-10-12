import subprocess
from unittest.mock import Mock, call

import pytest

from video_ingestion import recording_worker


@pytest.fixture
def retry_context(tmp_path, monkeypatch):
    entry = recording_worker.SegmentEntry("000000.mp4.part", 12.0, 18.5)
    segment = recording_worker.FinalizedSegment(
        tmp_path / "000000.mp4", 12.0, 18.5, 6.0
    )
    finalize = Mock(return_value=segment)
    sleep = Mock()
    monkeypatch.setattr(recording_worker, "_finalize_segment", finalize)
    monkeypatch.setattr(recording_worker.time, "sleep", sleep)
    return entry, segment, finalize, sleep


def test_finalize_segment_with_retry_succeeds_without_sleeping(retry_context, tmp_path):
    entry, segment, finalize, sleep = retry_context

    result = recording_worker._finalize_segment_with_retry(entry, tmp_path)

    assert result is segment
    finalize.assert_called_once_with(entry, tmp_path)
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [PermissionError("File locked"), subprocess.CalledProcessError(1, "ffprobe")],
    ids=["permission-error", "probe-error"],
)
def test_finalize_segment_with_retry_succeeds_on_last_attempt(
    retry_context, tmp_path, error
):
    entry, segment, finalize, sleep = retry_context
    events = Mock()
    events.attach_mock(finalize, "finalize")
    events.attach_mock(sleep, "sleep")
    finalize.side_effect = [error, error, segment]

    result = recording_worker._finalize_segment_with_retry(entry, tmp_path)

    assert result is segment
    assert events.mock_calls == [
        call.finalize(entry, tmp_path),
        call.sleep(0.25),
        call.finalize(entry, tmp_path),
        call.sleep(0.25),
        call.finalize(entry, tmp_path),
    ]


@pytest.mark.parametrize("max_attempts", [1, 3, 5])
@pytest.mark.parametrize(
    "error",
    [PermissionError("File locked"), subprocess.CalledProcessError(1, "ffprobe")],
    ids=["permission-error", "probe-error"],
)
def test_finalize_segment_with_retry_reraises_after_attempt_limit(
    retry_context, tmp_path, error, max_attempts
):
    entry, _, finalize, sleep = retry_context
    finalize.side_effect = error

    with pytest.raises(type(error)) as exc_info:
        recording_worker._finalize_segment_with_retry(
            entry, tmp_path, max_attempts=max_attempts
        )

    assert exc_info.value is error
    assert finalize.call_args_list == [call(entry, tmp_path)] * max_attempts
    assert sleep.call_args_list == [call(0.25)] * (max_attempts - 1)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Invalid segment duration"),
        FileExistsError("Destination exists"),
        FileNotFoundError("Segment missing"),
        subprocess.TimeoutExpired("ffprobe", 10),
    ],
    ids=["invalid-duration", "destination-exists", "missing-file", "timeout"],
)
def test_finalize_segment_with_retry_does_not_retry_other_errors(
    retry_context, tmp_path, error
):
    entry, _, finalize, sleep = retry_context
    finalize.side_effect = error

    with pytest.raises(type(error)) as exc_info:
        recording_worker._finalize_segment_with_retry(entry, tmp_path)

    assert exc_info.value is error
    finalize.assert_called_once_with(entry, tmp_path)
    sleep.assert_not_called()
