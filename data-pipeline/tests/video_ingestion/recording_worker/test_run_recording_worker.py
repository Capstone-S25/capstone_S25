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
    probe = Mock(return_value=6.0)
    monkeypatch.setattr(recording_worker, "_probe_segment_duration", probe)
    return process, session_dir, probe


def test_probes_running_entries_once_and_cleans_up(
    worker_process, make_source, tmp_path, monkeypatch
):
    process, session_dir, probe = worker_process
    csv_path = session_dir / "completed.csv"
    csv_path.write_text("000000.mp4.part,0,1\n", encoding="utf-8")
    pauses = 0

    def advance_recording(_seconds):
        nonlocal pauses
        pauses += 1
        if pauses == 1:
            # The first row was consumed while the process was still running
            probe.assert_called_once_with(session_dir / "000000.mp4.part")
            with csv_path.open("a", encoding="utf-8") as file:
                file.write("000001.mp4.part,1,2\n")
        else:
            process.poll.return_value = 0
            process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", advance_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    assert probe.call_args_list == [
        call(session_dir / "000000.mp4.part"),
        call(session_dir / "000001.mp4.part"),
    ]
    assert pauses == 2
    process.kill.assert_not_called()
    process.communicate.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_probes_all_rows_in_order_without_duplicates(
    worker_process, make_source, tmp_path, monkeypatch
):
    process, session_dir, probe = worker_process
    (session_dir / "completed.csv").write_text(
        "000000.mp4.part,0.000000,8.366667\n"
        "000001.mp4.part,8.366667,12.533333\n"
        "000002.mp4.part,12.533333,15.100000\n",
        encoding="utf-8",
    )
    expected_calls = [
        call(session_dir / "000000.mp4.part"),
        call(session_dir / "000001.mp4.part"),
        call(session_dir / "000002.mp4.part"),
    ]

    def finish_recording(_seconds):
        # All three segments must be probed before the process exits
        assert probe.call_args_list == expected_calls
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", finish_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    # The final read must not repeat rows already consumed by the loop
    assert probe.call_args_list == expected_calls
    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()


def test_probes_entries_written_at_process_exit(
    worker_process, make_source, tmp_path, monkeypatch
):
    """Probe the final CSV row written between the last read and process exit."""
    process, session_dir, probe = worker_process

    def finish_recording(_seconds):
        # Called during the loop's pause, after its first CSV read
        probe.assert_not_called()
        (session_dir / "completed.csv").write_text(
            "000000.mp4.part,0.000000,8.366667\n",
            encoding="utf-8",
        )

        # FFmpeg has now exited successfully
        process.poll.return_value = 0
        process.returncode = 0

    monkeypatch.setattr(recording_worker.time, "sleep", finish_recording)

    recording_worker.run_recording_worker(make_source(), tmp_path)

    probe.assert_called_once_with(session_dir / "000000.mp4.part")

    process.kill.assert_not_called()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()
