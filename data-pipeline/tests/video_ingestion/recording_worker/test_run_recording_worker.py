from unittest.mock import Mock

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
    return process, session_dir


def test_prints_running_entries_once_and_cleans_up(
    worker_process, make_source, tmp_path, monkeypatch, capsys
):
    process, session_dir = worker_process
    csv_path = session_dir / "completed.csv"
    csv_path.write_text("000000.mp4.part,0,1\n", encoding="utf-8")
    pauses = 0

    def advance_recording(_seconds):
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            # The first row was consumed while the process was still running
            assert capsys.readouterr().out.count("000000.mp4.part") == 1
            with csv_path.open("a", encoding="utf-8") as file:
                file.write("000001.mp4.part,1,2\n")
        else:
            process.poll.return_value = 0
            process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", advance_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    output = capsys.readouterr().out
    assert "000000.mp4.part" not in output
    assert output.count("000001.mp4.part") == 1
    assert pauses == 2
    process.kill.assert_not_called()
    process.communicate.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_prints_all_rows_in_order_without_duplicates(
    worker_process, make_source, tmp_path, monkeypatch, capsys
):
    process, session_dir = worker_process
    (session_dir / "completed.csv").write_text(
        "000000.mp4.part,0.000000,8.366667\n"
        "000001.mp4.part,8.366667,12.533333\n"
        "000002.mp4.part,12.533333,15.100000\n",
        encoding="utf-8",
    )
    expected_lines = [
        "SegmentEntry(filename='000000.mp4.part', "
        "start_seconds=0.0, end_seconds=8.366667)",
        "SegmentEntry(filename='000001.mp4.part', "
        "start_seconds=8.366667, end_seconds=12.533333)",
        "SegmentEntry(filename='000002.mp4.part', "
        "start_seconds=12.533333, end_seconds=15.1)",
    ]

    def finish_recording(_seconds):
        # All three rows must be printed before the process exits
        assert capsys.readouterr().out.splitlines() == expected_lines
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", finish_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    # The final read must not repeat rows already consumed by the loop
    assert capsys.readouterr().out == ""
    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_prints_entries_written_at_process_exit(
    worker_process, make_source, tmp_path, monkeypatch, capsys
):
    """
    1. The mock process is running, so the worker enters the loop
    2. The first CSV read finds no entries
    3. Instead of sleeping, finish_recording() writes a CSV row and marks the process as exited
    4. The next poll() returns 0, so the loop ends
    5. The worker's final CSV read should find and print that row
    """
    process, session_dir = worker_process

    def finish_recording(_seconds):
        # Called during the loop's pause, after its first CSV read
        (session_dir / "completed.csv").write_text(
            "000000.mp4.part,0.000000,8.366667\n",
            encoding="utf-8",
        )

        # FFmpeg has now exited successfully
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", finish_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    output = capsys.readouterr().out

    assert output.count("000000.mp4.part") == 1
    assert "start_seconds=0.0" in output
    assert "end_seconds=8.366667" in output

    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()
