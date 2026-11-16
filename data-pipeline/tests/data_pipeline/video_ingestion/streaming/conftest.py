from fractions import Fraction
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import av
import numpy as np
import pytest

from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming import worker


@pytest.fixture
def make_frame():
    """Use real PyAV pixels while tracking which frames are converted."""

    def factory(pts=0):
        pixels = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        return SimpleNamespace(
            pts=pts,
            time_base=Fraction(1, 1000),
            is_corrupt=False,
            to_ndarray=Mock(wraps=frame.to_ndarray),
        )

    return factory


@pytest.fixture
def stream(monkeypatch, make_frame):
    video = MagicMock()
    video.streams.video = [object()]
    video.decode.return_value = iter([make_frame()])
    video.__enter__.return_value = video

    def close_video(*_exception):
        video.close()
        return False

    video.__exit__.side_effect = close_video
    open_video = Mock(return_value=video)
    monkeypatch.setattr(worker.av, "open", open_video)
    clock = SimpleNamespace(
        monotonic=Mock(return_value=100.0),
        time_ns=Mock(return_value=1_700_000_000_000_000_000),
    )
    monkeypatch.setattr(worker, "time", clock)
    stop = Event()
    # Check playback delays without actually sleeping.
    monkeypatch.setattr(stop, "wait", Mock(return_value=False))
    context = SimpleNamespace(
        source=VideoSource("camera_01", "Test camera", SourceType.FILE, "clip.mp4"),
        video=video,
        open=open_video,
        clock=clock,
        queue=Queue(),
        stop=stop,
        done=Event(),
    )

    def run(**kwargs):
        worker.run_streaming_worker(
            context.source,
            context.queue,
            stop_event=context.stop,
            producer_done=context.done,
            **kwargs,
        )

    context.run = run
    return context
