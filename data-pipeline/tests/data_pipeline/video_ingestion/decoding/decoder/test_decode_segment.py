from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from av.error import InvalidDataError

from data_pipeline.video_ingestion.decoding.decoder import decode_segment


def test_yields_all_frames_with_metadata_indices_and_bgr_pixels(video, make_frame):
    video.container.decode.return_value = iter(
        [make_frame(0), make_frame(40), make_frame(80)]
    )

    frames = list(decode_segment(video.job, video.root))

    assert len(frames) == 3
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    for frame in frames:
        assert frame.segment_id == video.job.segment_id
        assert frame.source_id == video.job.source_id
        assert frame.session_id == video.job.session_id
        assert frame.image.shape == (4, 6, 3)
        assert frame.image.dtype == np.uint8
        # RGB input [10, 20, 30] must become BGR, not merely retain its shape.
        expected = np.empty((4, 6, 3), dtype=np.uint8)
        expected[:] = [30, 20, 10]
        np.testing.assert_array_equal(frame.image, expected)
    video.open.assert_called_once_with(str(video.path.resolve()), mode="r")
    video.container.decode.assert_called_once_with(video.stream)
    video.container.close.assert_called_once()


def test_timestamps_preserve_initial_pts_and_uneven_intervals(video, make_frame):
    # The first PTS is nonzero, and gaps are 40 ms then 90 ms.
    video.container.decode.return_value = iter(
        [
            make_frame(500),
            make_frame(540),
            make_frame(630),
        ]
    )

    frames = list(decode_segment(video.job, video.root))

    assert [frame.timestamp_ms for frame in frames] == [
        video.job.start_at_ms + 500,
        video.job.start_at_ms + 540,
        video.job.start_at_ms + 630,
    ]


def test_timestamps_remain_continuous_across_recorded_segments(video, make_frame):
    # First clip's CSV interval is 0-12 s, but its pictures occupy 2-12 s.
    video.container.decode.return_value = iter([make_frame(2000), make_frame(11960)])
    first_clip = list(decode_segment(video.job, video.root))

    # The next CSV interval starts at 12 s and its local PTS restarts at zero.
    next_job = replace(
        video.job, segment_id=43, start_at_ms=video.job.start_at_ms + 12000
    )
    video.container.decode.return_value = iter([make_frame(0), make_frame(40)])
    next_clip = list(decode_segment(next_job, video.root))

    assert first_clip[0].timestamp_ms == video.job.start_at_ms + 2000
    assert first_clip[-1].timestamp_ms == video.job.start_at_ms + 11960
    assert next_clip[0].timestamp_ms == video.job.start_at_ms + 12000
    assert next_clip[1].timestamp_ms == video.job.start_at_ms + 12040
    assert next_clip[0].timestamp_ms - first_clip[-1].timestamp_ms == 40


def test_decreasing_pts_raises_and_closes_container(video, make_frame):
    video.container.decode.return_value = iter([make_frame(100), make_frame(99)])
    frames = decode_segment(video.job, video.root)
    assert next(frames).frame_index == 0

    with pytest.raises(ValueError, match="Decreasing presentation"):
        next(frames)

    video.container.close.assert_called_once()


def test_rejects_absolute_path_before_opening_video(video):
    job = replace(video.job, path=str(video.path.resolve()))

    with pytest.raises(ValueError, match="must be relative"):
        list(decode_segment(job, video.root))

    video.open.assert_not_called()


def test_rejects_path_escaping_recordings_root(video):
    # Use an existing target so strict resolution succeeds before containment checking.
    outside = video.root.parent / "outside.mp4"
    outside.write_bytes(b"unrelated video")
    job = replace(video.job, path="../outside.mp4")

    with pytest.raises(ValueError, match="outside the recordings root"):
        list(decode_segment(job, video.root))

    video.open.assert_not_called()


def test_rejects_symlink_pointing_outside_root(video):
    outside = video.root.parent / "outside.mp4"
    outside.write_bytes(b"unrelated video")
    link = video.root / "link.mp4"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as error:
        if (
            isinstance(error, NotImplementedError)
            or getattr(error, "winerror", None) == 1314
        ):
            pytest.skip(
                "Creating symlinks requires platform support or Windows permission"
            )
        raise

    with pytest.raises(ValueError, match="outside the recordings root"):
        list(decode_segment(replace(video.job, path=link.name), video.root))

    video.open.assert_not_called()


def test_missing_file_raises_before_opening_video(video):
    video.path.unlink()

    with pytest.raises(FileNotFoundError):
        list(decode_segment(video.job, video.root))

    video.open.assert_not_called()


def test_directory_instead_of_file_is_rejected(video):
    directory = video.root / "not-a-video"
    directory.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        list(decode_segment(replace(video.job, path=directory.name), video.root))

    video.open.assert_not_called()


@pytest.mark.parametrize(
    ("pts", "time_base"),
    [(None, Fraction(1, 1000)), (0, None), (0, Fraction(0)), (0, Fraction(-1, 1000))],
    ids=["missing-pts", "missing-time-base", "zero-time-base", "negative-time-base"],
)
def test_invalid_frame_timing_raises_and_closes_container(video, pts, time_base):
    # PyAV setters reject some invalid values, so simulate malformed frame metadata.
    frame = SimpleNamespace(pts=pts, time_base=time_base, is_corrupt=False)
    video.container.decode.return_value = iter([frame])

    with pytest.raises(ValueError, match="Missing or invalid timing"):
        list(decode_segment(video.job, video.root))

    video.container.close.assert_called_once()


def test_no_video_stream_raises_and_closes_container(video):
    video.container.streams.video = []

    with pytest.raises(ValueError, match="no video stream"):
        list(decode_segment(video.job, video.root))

    video.container.decode.assert_not_called()
    video.container.close.assert_called_once()


def test_zero_decoded_frames_raises_and_closes_container(video):
    video.container.decode.return_value = iter([])

    with pytest.raises(ValueError, match="no decoded frames"):
        list(decode_segment(video.job, video.root))

    video.container.close.assert_called_once()


def test_corrupt_frame_raises_and_closes_container(video):
    video.container.decode.return_value = iter([SimpleNamespace(is_corrupt=True)])

    with pytest.raises(ValueError, match="Corrupt video frame"):
        list(decode_segment(video.job, video.root))

    video.container.close.assert_called_once()


def test_pyav_decode_exception_propagates_and_closes_container(video, make_frame):
    error = InvalidDataError(1094995529, "Invalid video packet")

    def failing_decode():
        yield make_frame()
        raise error

    video.container.decode.return_value = failing_decode()
    frames = decode_segment(video.job, video.root)
    next(frames)
    with pytest.raises(InvalidDataError) as raised:
        next(frames)

    assert raised.value is error
    video.container.close.assert_called_once()


def test_explicit_close_after_one_frame_closes_container(video, make_frame):
    video.container.decode.return_value = iter([make_frame(0), make_frame(40)])
    frames = decode_segment(video.job, video.root)
    assert next(frames).frame_index == 0
    video.container.close.assert_not_called()

    frames.close()

    video.container.close.assert_called_once()
    with pytest.raises(StopIteration):
        next(frames)


def test_opening_error_propagates(video):
    error = InvalidDataError(1094995529, "Invalid video container")
    video.open.side_effect = error

    with pytest.raises(InvalidDataError) as raised:
        list(decode_segment(video.job, video.root))

    assert raised.value is error
    # No container was opened, so there is nothing to close.
    video.container.__enter__.assert_not_called()
