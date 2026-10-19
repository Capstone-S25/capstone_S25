from datetime import UTC, datetime, timedelta
from threading import Event
from unittest.mock import MagicMock, Mock, call

import pytest

from video_ingestion.recording import worker
from video_ingestion.recording.segments import FinalizedSegment, SegmentEntry


@pytest.fixture
def session_factory(monkeypatch):
    clock = Mock()
    clock.now.return_value = datetime(2026, 10, 17, 12, tzinfo=UTC)
    monkeypatch.setattr(worker, "datetime", clock)
    factory = MagicMock()
    monkeypatch.setattr(
        worker, "insert_video_segment_metadata", Mock(return_value=True)
    )
    return factory


@pytest.fixture
def worker_process(tmp_path, monkeypatch, session_factory):
    """A running process and session directory, without launching FFmpeg."""
    monkeypatch.setattr(worker.time, "time_ns", lambda: 1_700_000_000_000_999_999)
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    process = Mock()
    process.poll.return_value = None
    process.returncode = None
    process.stdin = Mock()
    monkeypatch.setattr(worker, "_prepare_session", lambda *_: ("session", session_dir))
    monkeypatch.setattr(worker, "start_ffmpeg", lambda _: process)
    # Return realistic metadata without renaming files; rename tests are separate.
    finalize = Mock(
        side_effect=lambda entry, directory, *, session_started_at_utc: (
            FinalizedSegment(
                path=(directory / entry.filename).with_suffix(""),
                start_at_ms=session_started_at_utc + round(entry.start_seconds * 1000),
                end_at_ms=session_started_at_utc + round(entry.end_seconds * 1000),
            )
        )
    )
    monkeypatch.setattr(worker, "finalize_segment_with_retry", finalize)
    return process, session_dir, finalize


def _assert_segment_metadata_registered(
    finalize, session_factory, storage_root, retention_hours=24
):
    """Check that each finalized segment is inserted once with correct metadata."""
    finalized_calls = finalize.call_args_list
    insert_calls = worker.insert_video_segment_metadata.call_args_list
    transaction = session_factory.begin.return_value
    db = transaction.__enter__.return_value
    expected_deadline = datetime(2026, 10, 17, 12, tzinfo=UTC) + timedelta(
        hours=retention_hours
    )

    assert len(insert_calls) == len(finalized_calls)

    # Pair each finalized CSV entry with its database insertion, in order.
    for finalized_call, insert_call in zip(finalized_calls, insert_calls, strict=True):
        entry, session_dir = finalized_call.args
        session_start_ms = finalized_call.kwargs["session_started_at_utc"]
        metadata = insert_call.kwargs

        assert insert_call.args == (db,)
        assert metadata["source_id"] == "camera_01"
        assert metadata["session_id"] == "session"
        assert metadata["storage_root"] == storage_root
        assert metadata["path"] == (session_dir / entry.filename).with_suffix("")
        assert metadata["start_at_ms"] == session_start_ms + round(
            entry.start_seconds * 1000
        )
        assert metadata["end_at_ms"] == session_start_ms + round(
            entry.end_seconds * 1000
        )
        assert metadata["delete_at"] == expected_deadline

    # Begin a separate transaction for each segment insertion.
    assert session_factory.begin.call_count == len(insert_calls)


@pytest.mark.parametrize("retention_hours", [24, 48, 0.5])
def test_finalizes_running_entries_once_and_cleans_up(
    worker_process, make_source, tmp_path, monkeypatch, session_factory, retention_hours
):
    process, session_dir, finalize = worker_process
    csv_path = session_dir / "completed.csv"
    csv_path.write_text("000000.mp4.part,0,1\n", encoding="utf-8")
    pauses = 0

    def advance_recording(_seconds):
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            # The first row was consumed while the process was still running
            finalize.assert_called_once_with(
                SegmentEntry("000000.mp4.part", 0.0, 1.0),
                session_dir,
                session_started_at_utc=1_700_000_000_000,
            )
            with csv_path.open("a", encoding="utf-8") as file:
                file.write("000001.mp4.part,1,2\n")
        else:
            process.poll.return_value = 0
            process.returncode = 0

    monkeypatch.setattr(worker.time, "sleep", advance_recording)

    worker.run_recording_worker(
        make_source(),
        tmp_path,
        session_factory=session_factory,
        retention_hours=retention_hours,
    )

    assert finalize.call_args_list == [
        call(
            SegmentEntry("000000.mp4.part", 0.0, 1.0),
            session_dir,
            session_started_at_utc=1_700_000_000_000,
        ),
        call(
            SegmentEntry("000001.mp4.part", 1.0, 2.0),
            session_dir,
            session_started_at_utc=1_700_000_000_000,
        ),
    ]
    assert pauses == 2
    _assert_segment_metadata_registered(
        finalize, session_factory, tmp_path, retention_hours
    )

    process.kill.assert_not_called()
    process.communicate.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_finalizes_all_rows_in_order_without_duplicates(
    worker_process, make_source, tmp_path, monkeypatch, session_factory
):
    process, session_dir, finalize = worker_process
    (session_dir / "completed.csv").write_text(
        "000000.mp4.part,0.000000,8.366667\n"
        "000001.mp4.part,8.366667,12.533333\n"
        "000002.mp4.part,12.533333,15.100000\n",
        encoding="utf-8",
    )
    expected_calls = [
        call(
            SegmentEntry("000000.mp4.part", 0.0, 8.366667),
            session_dir,
            session_started_at_utc=1_700_000_000_000,
        ),
        call(
            SegmentEntry("000001.mp4.part", 8.366667, 12.533333),
            session_dir,
            session_started_at_utc=1_700_000_000_000,
        ),
        call(
            SegmentEntry("000002.mp4.part", 12.533333, 15.1),
            session_dir,
            session_started_at_utc=1_700_000_000_000,
        ),
    ]

    def finish_recording(_seconds):
        # All three segments must be finalized before the process exits
        assert finalize.call_args_list == expected_calls
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(worker.time, "sleep", finish_recording)

    worker.run_recording_worker(
        make_source(), tmp_path, session_factory=session_factory
    )

    # The final read must not repeat rows already consumed by the loop
    assert finalize.call_args_list == expected_calls
    _assert_segment_metadata_registered(finalize, session_factory, tmp_path)

    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


@pytest.mark.parametrize("retention_hours", [24, 48, 0.5])
def test_finalizes_entries_written_at_process_exit(
    worker_process, make_source, tmp_path, monkeypatch, session_factory, retention_hours
):
    """Finalize the final CSV row written between the last read and process exit."""
    process, session_dir, finalize = worker_process

    def finish_recording(_seconds):
        # Called during the loop's pause, after its first CSV read
        finalize.assert_not_called()
        (session_dir / "completed.csv").write_text(
            "000000.mp4.part,0.000000,8.366667\n",
            encoding="utf-8",
        )

        # FFmpeg has now exited successfully
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(worker.time, "sleep", finish_recording)

    worker.run_recording_worker(
        make_source(),
        tmp_path,
        session_factory=session_factory,
        retention_hours=retention_hours,
    )

    finalize.assert_called_once_with(
        SegmentEntry("000000.mp4.part", 0.0, 8.366667),
        session_dir,
        session_started_at_utc=1_700_000_000_000,
    )

    _assert_segment_metadata_registered(
        finalize, session_factory, tmp_path, retention_hours
    )

    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_exits_gracefully_when_stop_event_is_set(
    worker_process, make_source, tmp_path, session_factory, monkeypatch
):
    process, session_dir, finalize = worker_process
    stop_event = Event()
    # 1. Start with one completed segment while FFmpeg is still running.
    csv_path = session_dir / "completed.csv"
    csv_path.write_text("000000.mp4.part,0.0,5.0\n", encoding="utf-8")

    def request_stop(_seconds):
        # 2. During the loop's pause, request shutdown after the first segment.
        finalize.assert_called_once()
        stop_event.set()

    def close_recording(**kwargs):
        # 3. Simulate FFmpeg writing its last segment when it receives "q\n".
        with csv_path.open("a", encoding="utf-8") as file:
            file.write("000001.mp4.part,5.0,7.5\n")
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(worker.time, "sleep", request_stop)
    # Calling the mocked communicate() runs close_recording() instead of FFmpeg.
    process.communicate.side_effect = close_recording

    worker.run_recording_worker(
        make_source(), tmp_path, session_factory=session_factory, stop_event=stop_event
    )

    # 4. Verify graceful shutdown and cleanup, without force-killing FFmpeg.
    assert stop_event.is_set()
    process.communicate.assert_called_once_with(input="q\n", timeout=10)
    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()

    # 5. Both the initial and final partial segments must be finalized and registered.
    finalized_entries = [call.args[0] for call in finalize.call_args_list]
    assert finalized_entries == [
        SegmentEntry("000000.mp4.part", 0.0, 5.0),
        SegmentEntry("000001.mp4.part", 5.0, 7.5),
    ]
    _assert_segment_metadata_registered(finalize, session_factory, tmp_path)
