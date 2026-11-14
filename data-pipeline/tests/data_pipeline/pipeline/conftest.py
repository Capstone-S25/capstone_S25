from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from data_pipeline.pipeline import runner
from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming.worker import StreamFrame

WORKER_WAIT_SECONDS = 5


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Run the real orchestrator and threads with controllable video workers."""
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"worker input is stubbed")
    context = SimpleNamespace(
        source=VideoSource(
            "camera_01", "Test camera", SourceType.FILE, str(source_path)
        ),
        session_factory=Mock(),
        recording=Mock(return_value=None),
        recording_started=Event(),
        recording_finished=Event(),
        streaming_finished=Event(),
        frames=[
            StreamFrame(
                source_id="camera_01",
                frame_index=index,
                media_timestamp_ms=index * 500,
                timestamp_ms=1_700_000_000_000 + index * 500,
                image=np.zeros((2, 3, 3), dtype=np.uint8),
            )
            for index in range(8)
        ],
    )

    def enqueue(_source, queue, **_kwargs):
        for frame in context.frames:
            queue.put_nowait(frame)

    context.streaming = Mock(side_effect=enqueue)

    def record(*args, **kwargs):
        context.recording_started.set()
        try:
            return context.recording(*args, **kwargs)
        finally:
            context.recording_finished.set()

    def stream(source, queue, **kwargs):
        context.queue = queue
        context.stop = kwargs["stop_event"]
        context.done = kwargs["producer_done"]
        try:
            return context.streaming(source, queue, **kwargs)
        finally:
            context.done.set()
            context.streaming_finished.set()

    monkeypatch.setattr(runner, "run_recording_worker", record)
    monkeypatch.setattr(runner, "run_streaming_worker", stream)
    monkeypatch.setattr(runner, "FRAME_QUEUE_TIMEOUT_SECONDS", 0.005)

    def run(handler, **kwargs):
        # Lifecycle scenarios keep their finite backlog within queue capacity.
        kwargs.setdefault("queue_size", len(context.frames))
        runner.run_pipeline(
            context.source,
            tmp_path / "recordings",
            session_factory=context.session_factory,
            frame_handler=handler,
            **kwargs,
        )

    context.run = run
    context.wait_seconds = WORKER_WAIT_SECONDS
    return context
