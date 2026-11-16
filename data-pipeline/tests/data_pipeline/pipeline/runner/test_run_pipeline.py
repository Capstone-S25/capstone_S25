import logging
from dataclasses import replace
from threading import Event, get_ident
from unittest.mock import Mock

import pytest


def test_consumes_fifo_backlog_before_recording_finishes_and_waits_for_both_workers(
    pipeline,
):
    finish_recording = Event()
    received = []
    caller_thread = get_ident()

    def record(*_args, **_kwargs):
        assert finish_recording.wait(pipeline.wait_seconds)

    def consume(frame):
        assert get_ident() == caller_thread
        if not received:
            assert pipeline.recording_started.wait(pipeline.wait_seconds)
            # Hold the consumer until the producer has built a complete backlog.
            assert pipeline.done.wait(pipeline.wait_seconds)
            assert not pipeline.recording_finished.is_set()
            assert pipeline.queue.maxsize == len(pipeline.frames)
            assert pipeline.queue.qsize() == len(pipeline.frames) - 1
        received.append(frame.frame_index)
        if len(received) == len(pipeline.frames):
            finish_recording.set()

    pipeline.recording.side_effect = record
    pipeline.run(consume)

    assert received == list(range(len(pipeline.frames)))
    assert pipeline.queue.empty()
    assert pipeline.recording_finished.is_set()
    assert pipeline.streaming_finished.is_set()
    # Live consumption must not claim or update recorded segments in the DB.
    pipeline.session_factory.begin.assert_not_called()


def test_ctrl_c_stops_workers_retries_interrupted_frame_and_drains_queue(pipeline):
    received = []
    interrupted = False

    def record(*_args, stop_event, **_kwargs):
        assert stop_event.wait(pipeline.wait_seconds)

    def consume(frame):
        nonlocal interrupted
        if not interrupted:
            assert pipeline.done.wait(pipeline.wait_seconds)
            interrupted = True
            raise KeyboardInterrupt
        received.append(frame.frame_index)

    pipeline.recording.side_effect = record
    pipeline.run(consume)

    assert received == list(range(len(pipeline.frames)))
    assert pipeline.stop.is_set()
    assert pipeline.queue.empty()
    assert pipeline.recording_finished.is_set()
    assert pipeline.streaming_finished.is_set()


@pytest.mark.parametrize("failing_worker", ["recording", "streaming"])
def test_worker_failure_stops_peer_and_drains_accepted_frames(pipeline, failing_worker):
    enqueued = Event()
    error = RuntimeError(f"{failing_worker} failed")
    received = []

    def record(*_args, stop_event, **_kwargs):
        if failing_worker == "recording":
            assert enqueued.wait(pipeline.wait_seconds)
            raise error
        assert stop_event.wait(pipeline.wait_seconds)

    def stream(_source, queue, stop_event, **_kwargs):
        for frame in pipeline.frames:
            queue.put_nowait(frame)
        enqueued.set()
        if failing_worker == "streaming":
            raise error
        assert stop_event.wait(pipeline.wait_seconds)

    pipeline.recording.side_effect = record
    pipeline.streaming.side_effect = stream

    with pytest.raises(RuntimeError) as raised:
        pipeline.run(lambda frame: received.append(frame.frame_index))

    assert raised.value is error
    assert received == list(range(len(pipeline.frames)))
    assert pipeline.queue.empty()
    assert pipeline.stop.is_set()
    assert pipeline.recording_finished.is_set()
    assert pipeline.streaming_finished.is_set()


def test_consumer_failure_stops_callbacks_and_preserves_error_during_cleanup(
    pipeline, caplog
):
    error = RuntimeError("Consumer failed")
    cleanup_error = RuntimeError("Recorder cleanup failed")
    consume = Mock(side_effect=error)

    def record(*_args, stop_event, **_kwargs):
        assert stop_event.wait(pipeline.wait_seconds)
        raise cleanup_error

    pipeline.recording.side_effect = record
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as raised:
        pipeline.run(consume)

    assert raised.value is error
    consume.assert_called_once()
    assert pipeline.stop.is_set()
    assert pipeline.recording_finished.is_set()
    assert pipeline.streaming_finished.is_set()
    cleanup_logs = [r for r in caplog.records if r.exc_info]
    assert len(cleanup_logs) == 1
    assert cleanup_logs[0].exc_info[1] is cleanup_error


@pytest.mark.parametrize(
    "settings",
    [
        {"target_fps": 0},
        {"segment_seconds": -1},
        {"retention_hours": float("nan")},
    ],
)
def test_invalid_settings_do_not_start_either_worker(pipeline, settings):
    with pytest.raises(ValueError):
        pipeline.run(Mock(), **settings)

    pipeline.recording.assert_not_called()
    pipeline.streaming.assert_not_called()


def test_missing_file_does_not_start_either_worker(pipeline):
    pipeline.source = replace(
        pipeline.source, location=pipeline.source.location + ".missing"
    )

    with pytest.raises(FileNotFoundError):
        pipeline.run(Mock())

    pipeline.recording.assert_not_called()
    pipeline.streaming.assert_not_called()
