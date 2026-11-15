import logging
from dataclasses import replace
from threading import Event, get_ident
from unittest.mock import Mock

import pytest

from data_pipeline.pipeline import runner


def test_round_robin_preserves_fifo_and_uses_one_consumer_for_four_cameras(
    multi_camera_pipeline, monkeypatch
):
    pipeline = multi_camera_pipeline
    finish_recording = Event()
    caller_thread = get_ident()
    received = []
    expected = [
        (source.source_id, index) for index in range(3) for source in pipeline.sources
    ]
    start_workers = runner._start_workers

    def start_with_ready_queues(*args, **kwargs):
        start_workers(*args, **kwargs)
        # All queues must have a backlog before round-robin order can be asserted.
        for finished in pipeline.streaming_finished.values():
            assert finished.wait(pipeline.wait_seconds)

    def record(*_args, **_kwargs):
        assert finish_recording.wait(pipeline.wait_seconds)

    def consume(frame):
        assert get_ident() == caller_thread
        if not received:
            assert not any(e.is_set() for e in pipeline.recording_finished.values())
        received.append((frame.source_id, frame.frame_index))
        if len(received) == len(expected):
            finish_recording.set()

    monkeypatch.setattr(runner, "_start_workers", start_with_ready_queues)
    pipeline.recording.side_effect = record
    pipeline.run(consume)

    assert received == expected
    assert all(e.is_set() for e in pipeline.recording_finished.values())
    assert all(queue.empty() for queue in pipeline.queues.values())


def test_skips_empty_camera_then_returns_to_it_after_another_camera_reaches_eof(
    multi_camera_pipeline,
):
    pipeline = multi_camera_pipeline
    pipeline.sources = pipeline.sources[:2]
    delayed, ready = [source.source_id for source in pipeline.sources]
    pipeline.frames[delayed] = pipeline.frames[delayed][:2]
    release_delayed_camera = Event()
    received = []

    def stream(source, queue, stop_event, **_kwargs):
        if source.source_id == delayed:
            assert release_delayed_camera.wait(pipeline.wait_seconds)
            assert not stop_event.is_set()
        for frame in pipeline.frames[source.source_id]:
            queue.put_nowait(frame)

    def consume(frame):
        if not received:
            assert frame.source_id == ready
            assert pipeline.streaming_finished[ready].wait(pipeline.wait_seconds)
            # EOF on the ready camera must not stop the camera still waiting for input.
            release_delayed_camera.set()
            assert pipeline.streaming_finished[delayed].wait(pipeline.wait_seconds)
        received.append((frame.source_id, frame.frame_index))

    pipeline.streaming.side_effect = stream
    pipeline.run(consume)

    assert received == [(ready, 0), (delayed, 0), (ready, 1), (delayed, 1), (ready, 2)]
    assert all(queue.empty() for queue in pipeline.queues.values())


def test_ctrl_c_stops_all_cameras_retries_the_frame_and_drains_every_queue(
    multi_camera_pipeline,
):
    pipeline = multi_camera_pipeline
    ready = {source.source_id: Event() for source in pipeline.sources}
    attempts, received = [], []

    def record(*_args, stop_event, **_kwargs):
        assert stop_event.wait(pipeline.wait_seconds)

    def stream(source, queue, stop_event, **_kwargs):
        for frame in pipeline.frames[source.source_id]:
            queue.put_nowait(frame)
        ready[source.source_id].set()
        assert stop_event.wait(pipeline.wait_seconds)

    def consume(frame):
        attempts.append((frame.source_id, frame.frame_index))
        if len(attempts) == 1:
            for event in ready.values():
                assert event.wait(pipeline.wait_seconds)
            raise KeyboardInterrupt
        received.append((frame.source_id, frame.frame_index))

    pipeline.recording.side_effect = record
    pipeline.streaming.side_effect = stream
    pipeline.run(consume)

    assert attempts[0] == attempts[1]
    assert sorted(received) == sorted(
        (frame.source_id, frame.frame_index)
        for frames in pipeline.frames.values()
        for frame in frames
    )
    assert all(event.is_set() for event in pipeline.stop_events.values())
    assert all(event.is_set() for event in pipeline.recording_finished.values())
    assert all(event.is_set() for event in pipeline.streaming_finished.values())
    assert all(queue.empty() for queue in pipeline.queues.values())


@pytest.mark.parametrize("failing_worker", ["recording", "streaming"])
def test_worker_failure_stops_all_cameras_and_drains_their_accepted_frames(
    multi_camera_pipeline, failing_worker, caplog
):
    pipeline = multi_camera_pipeline
    ready = {source.source_id: Event() for source in pipeline.sources}
    failing_source = pipeline.sources[1].source_id
    error = RuntimeError(f"{failing_worker} failed")
    received = []

    def fail_after_all_cameras_enqueue():
        for event in ready.values():
            assert event.wait(pipeline.wait_seconds)
        raise error

    def record(source, *_args, stop_event, **_kwargs):
        if source.source_id == failing_source and failing_worker == "recording":
            fail_after_all_cameras_enqueue()
        assert stop_event.wait(pipeline.wait_seconds)

    def stream(source, queue, stop_event, **_kwargs):
        for frame in pipeline.frames[source.source_id]:
            queue.put_nowait(frame)
        ready[source.source_id].set()
        if source.source_id == failing_source and failing_worker == "streaming":
            fail_after_all_cameras_enqueue()
        assert stop_event.wait(pipeline.wait_seconds)

    pipeline.recording.side_effect = record
    pipeline.streaming.side_effect = stream
    with pytest.raises(RuntimeError) as raised:
        pipeline.run(
            lambda frame: received.append((frame.source_id, frame.frame_index))
        )

    assert raised.value is error
    assert sorted(received) == sorted(
        (frame.source_id, frame.frame_index)
        for frames in pipeline.frames.values()
        for frame in frames
    )
    assert all(event.is_set() for event in pipeline.stop_events.values())
    assert all(event.is_set() for event in pipeline.recording_finished.values())
    assert all(event.is_set() for event in pipeline.streaming_finished.values())
    assert all(queue.empty() for queue in pipeline.queues.values())
    # Background assertion failures must not be hidden behind the expected error.
    assert not [record for record in caplog.records if record.exc_info]


def test_consumer_failure_stops_every_camera_and_preserves_error_during_cleanup(
    multi_camera_pipeline, caplog
):
    pipeline = multi_camera_pipeline
    ready = {source.source_id: Event() for source in pipeline.sources}
    error = RuntimeError("Consumer failed")
    cleanup_error = RuntimeError("Recorder cleanup failed")

    def record(source, *_args, stop_event, **_kwargs):
        assert stop_event.wait(pipeline.wait_seconds)
        if source.source_id == pipeline.sources[2].source_id:
            raise cleanup_error

    def stream(source, queue, stop_event, **_kwargs):
        for frame in pipeline.frames[source.source_id]:
            queue.put_nowait(frame)
        ready[source.source_id].set()
        assert stop_event.wait(pipeline.wait_seconds)

    def fail(_frame):
        for event in ready.values():
            assert event.wait(pipeline.wait_seconds)
        raise error

    consume = Mock(side_effect=fail)
    pipeline.recording.side_effect = record
    pipeline.streaming.side_effect = stream
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as raised:
        pipeline.run(consume)

    assert raised.value is error
    consume.assert_called_once()
    assert all(event.is_set() for event in pipeline.stop_events.values())
    assert all(event.is_set() for event in pipeline.recording_finished.values())
    assert all(event.is_set() for event in pipeline.streaming_finished.values())
    assert any(not queue.empty() for queue in pipeline.queues.values())
    assert [record.exc_info[1] for record in caplog.records if record.exc_info] == [
        cleanup_error
    ]


@pytest.mark.parametrize("invalid_source", ["duplicate_id", "missing_file"])
def test_invalid_later_camera_prevents_all_workers_from_starting(
    multi_camera_pipeline, invalid_source
):
    pipeline = multi_camera_pipeline
    if invalid_source == "duplicate_id":
        pipeline.sources[-1] = pipeline.sources[0]
        error = ValueError
    else:
        pipeline.sources[-1] = replace(
            pipeline.sources[-1], location=pipeline.sources[-1].location + ".missing"
        )
        error = FileNotFoundError

    with pytest.raises(error):
        pipeline.run(Mock())

    pipeline.recording.assert_not_called()
    pipeline.streaming.assert_not_called()
