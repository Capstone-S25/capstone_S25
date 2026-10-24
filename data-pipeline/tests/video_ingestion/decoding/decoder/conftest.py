from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import av
import numpy as np
import pytest

from video_ingestion.decoding import decoder
from video_ingestion.scheduling.worker import SegmentJob


@pytest.fixture
def make_frame():
    """Create real PyAV frames so pixel conversion is exercised, not mocked."""

    def factory(pts=0, time_base=Fraction(1, 1000)):
        rgb = np.empty((4, 6, 3), dtype=np.uint8)
        rgb[:] = [10, 20, 30]
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    return factory


@pytest.fixture
def video(tmp_path, monkeypatch, make_frame):
    """Use a real path and fake container; no encoded video file is required."""
    root = tmp_path / "recordings"
    root.mkdir()
    path = root / "clip.mp4"
    path.write_bytes(b"mock video")
    job = SegmentJob(
        42, "camera_01", uuid4(), "clip.mp4", 1_700_000_000_000, 1_700_000_005_000
    )
    container = MagicMock()
    stream = object()
    container.streams.video = [stream]
    container.decode.return_value = iter([make_frame()])
    container.__enter__.return_value = container

    def close_container(*_exception):
        container.close()
        return False  # Never suppress decoding errors.

    container.__exit__.side_effect = close_container
    open_video = Mock(return_value=container)
    monkeypatch.setattr(decoder.av, "open", open_video)
    return SimpleNamespace(
        root=root,
        path=path,
        job=job,
        container=container,
        open=open_video,
        stream=stream,
    )
