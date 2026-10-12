from unittest.mock import Mock, call

import pytest

from video_ingestion import recording_worker


@pytest.fixture
def worker_process(tmp_path, monkeypatch):
    """A running process and session directory, without launching FFmpeg."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    process = Mock()
    process.poll.return_value = None
    process.returncode = None
    process.stdin = Mock()
    monkeypatch.setattr(
        recording_worker, "_prepare_session", lambda *_: ("session", session_dir)
    )
    monkeypatch.setattr(recording_worker, "_start_ffmpeg", lambda _: process)
    finalize = Mock(
        side_effect=lambda entry, directory: recording_worker.FinalizedSegment(
            path=(directory / entry.filename).with_suffix(""),
            start_seconds=entry.start_seconds,
            end_seconds=entry.end_seconds,
            duration_seconds=6.0,
        )
    )
    monkeypatch.setattr(recording_worker, "_finalize_segment_with_retry", finalize)
    return process, session_dir, finalize


def test_finalizes_running_entries_once_and_cleans_up(
    worker_process, make_source, tmp_path, monkeypatch
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
                recording_worker.SegmentEntry("000000.mp4.part", 0.0, 1.0),
                session_dir,
            )
            with csv_path.open("a", encoding="utf-8") as file:
                file.write("000001.mp4.part,1,2\n")
        else:
            process.poll.return_value = 0
            process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", advance_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    assert finalize.call_args_list == [
        call(recording_worker.SegmentEntry("000000.mp4.part", 0.0, 1.0), session_dir),
        call(recording_worker.SegmentEntry("000001.mp4.part", 1.0, 2.0), session_dir),
    ]
    assert pauses == 2
    process.kill.assert_not_called()
    process.communicate.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_finalizes_all_rows_in_order_without_duplicates(
    worker_process, make_source, tmp_path, monkeypatch
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
            recording_worker.SegmentEntry("000000.mp4.part", 0.0, 8.366667), session_dir
        ),
        call(
            recording_worker.SegmentEntry("000001.mp4.part", 8.366667, 12.533333),
            session_dir,
        ),
        call(
            recording_worker.SegmentEntry("000002.mp4.part", 12.533333, 15.1),
            session_dir,
        ),
    ]

    def finish_recording(_seconds):
        # All three segments must be finalized before the process exits
        assert finalize.call_args_list == expected_calls
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", finish_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    # The final read must not repeat rows already consumed by the loop
    assert finalize.call_args_list == expected_calls
    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_finalizes_entries_written_at_process_exit(
    worker_process, make_source, tmp_path, monkeypatch
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

    monkeypatch.setattr(recording_worker.time, "sleep", finish_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    finalize.assert_called_once_with(
        recording_worker.SegmentEntry("000000.mp4.part", 0.0, 8.366667),
        session_dir,
    )

    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()
