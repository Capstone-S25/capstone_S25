from unittest.mock import Mock, call

import pytest

from data_pipeline.video_ingestion.recording import segments
from data_pipeline.video_ingestion.recording.segments import (
    FinalizedSegment,
    SegmentEntry,
)


@pytest.fixture
def retry_context(tmp_path, monkeypatch):
    entry = SegmentEntry("000000.mp4.part", 12.0, 18.5)
    segment = FinalizedSegment(
        tmp_path / "000000.mp4", 1_700_000_012_000, 1_700_000_018_500
    )
    finalize = Mock(return_value=segment)
    sleep = Mock()
    monkeypatch.setattr(segments, "_finalize_segment", finalize)
    monkeypatch.setattr(segments.time, "sleep", sleep)
    return entry, segment, finalize, sleep


def test_finalize_segment_with_retry_succeeds_without_sleeping(retry_context, tmp_path):
    entry, segment, finalize, sleep = retry_context

    result = segments.finalize_segment_with_retry(
        entry, tmp_path, session_started_at_utc=1_700_000_000_000
    )

    assert result is segment
    finalize.assert_called_once_with(
        entry, tmp_path, session_started_at_utc=1_700_000_000_000
    )
    sleep.assert_not_called()


def test_finalize_segment_with_retry_succeeds_on_last_attempt(retry_context, tmp_path):
    entry, segment, finalize, sleep = retry_context
    error = PermissionError("File locked")
    events = Mock()
    events.attach_mock(finalize, "finalize")
    events.attach_mock(sleep, "sleep")
    finalize.side_effect = [error, error, segment]

    result = segments.finalize_segment_with_retry(
        entry, tmp_path, session_started_at_utc=1_700_000_000_000
    )

    assert result is segment
    assert events.mock_calls == [
        call.finalize(entry, tmp_path, session_started_at_utc=1_700_000_000_000),
        call.sleep(0.25),
        call.finalize(entry, tmp_path, session_started_at_utc=1_700_000_000_000),
        call.sleep(0.25),
        call.finalize(entry, tmp_path, session_started_at_utc=1_700_000_000_000),
    ]


@pytest.mark.parametrize("max_attempts", [1, 3, 5])
def test_finalize_segment_with_retry_reraises_after_attempt_limit(
    retry_context, tmp_path, max_attempts
):
    entry, _, finalize, sleep = retry_context
    error = PermissionError("File locked")
    finalize.side_effect = error

    with pytest.raises(type(error)) as exc_info:
        segments.finalize_segment_with_retry(
            entry,
            tmp_path,
            max_attempts=max_attempts,
            session_started_at_utc=1_700_000_000_000,
        )

    assert exc_info.value is error
    assert (
        finalize.call_args_list
        == [call(entry, tmp_path, session_started_at_utc=1_700_000_000_000)]
        * max_attempts
    )
    assert sleep.call_args_list == [call(0.25)] * (max_attempts - 1)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Expected a temporary segment"),
        FileExistsError("Destination exists"),
        FileNotFoundError("Segment missing"),
    ],
    ids=["invalid-filename", "destination-exists", "missing-file"],
)
def test_finalize_segment_with_retry_does_not_retry_other_errors(
    retry_context, tmp_path, error
):
    entry, _, finalize, sleep = retry_context
    finalize.side_effect = error

    with pytest.raises(type(error)) as exc_info:
        segments.finalize_segment_with_retry(
            entry, tmp_path, session_started_at_utc=1_700_000_000_000
        )

    assert exc_info.value is error
    finalize.assert_called_once_with(
        entry, tmp_path, session_started_at_utc=1_700_000_000_000
    )
    sleep.assert_not_called()
