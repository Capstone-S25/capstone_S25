from dataclasses import replace

import numpy as np
import pytest

from data_pipeline.video_ingestion.sources import SourceType
from data_pipeline.video_ingestion.streaming import worker


def test_selects_frames_with_original_indices_timestamps_and_bgr_pixels(
    stream, make_frame
):
    decoded = [make_frame(pts) for pts in [5000, 5100, 5500, 5600, 6000]]
    stream.video.decode.return_value = iter(decoded)

    stream.run(target_fps=2)

    frames = [stream.queue.get_nowait() for _ in range(stream.queue.qsize())]
    assert [frame.frame_index for frame in frames] == [0, 2, 4]
    assert [frame.media_timestamp_ms for frame in frames] == [5000, 5500, 6000]
    assert [frame.timestamp_ms for frame in frames] == [
        1_700_000_000_000,
        1_700_000_000_500,
        1_700_000_001_000,
    ]
    for frame in frames:
        assert frame.source_id == "camera_01"
        assert frame.image.shape == (4, 6, 3)
        assert frame.image.dtype == np.uint8
        expected = np.full((4, 6, 3), [30, 20, 10], dtype=np.uint8)
        np.testing.assert_array_equal(frame.image, expected)
    # Frames skipped by FPS selection must not incur pixel conversion.
    assert [frame.to_ndarray.call_count for frame in decoded] == [1, 0, 1, 0, 1]
    assert stream.done.is_set()
    stream.video.close.assert_called_once()


def test_rtsp_uses_tcp_timeout_and_no_pacing_and_reports_unexpected_eof(stream):
    stream.source = replace(
        stream.source, source_type=SourceType.RTSP, location="rtsp://camera/stream"
    )

    with pytest.raises(RuntimeError, match="RTSP stream ended unexpectedly"):
        stream.run()

    stream.open.assert_called_once_with(
        stream.source.location,
        mode="r",
        options={"rtsp_transport": "tcp"},
        timeout=worker.RTSP_TIMEOUT_SECONDS,
    )
    stream.stop.wait.assert_not_called()
    assert stream.queue.qsize() == 1
    assert stream.done.is_set()
    stream.video.close.assert_called_once()
