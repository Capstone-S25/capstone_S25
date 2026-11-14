import logging
import time
from dataclasses import dataclass
from fractions import Fraction
from threading import Event
from typing import cast

import av
import numpy as np
from av.container import InputContainer
from av.error import FFmpegError
from av.video.frame import VideoFrame
from numpy.typing import NDArray

from data_pipeline.video_ingestion.decoding.selection import FrameSelector
from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming.frame_queue import DropOldestQueue

logger = logging.getLogger(__name__)

RTSP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class StreamFrame:
    """A live frame selected by a streaming worker

    Attributes:
        source_id          : Camera or file source identifier
        frame_index        : Zero-based index counting every decoded frame, including
                             frames skipped by FPS selection
        media_timestamp_ms : Original presentation timestamp in milliseconds
        timestamp_ms       : Approximate UTC epoch milliseconds, anchored to receipt
                             of the first decoded frame and advanced by media time
        image              : Original-resolution BGR uint8 array with shape (height, width, 3)
    """

    source_id: str
    frame_index: int
    media_timestamp_ms: int
    timestamp_ms: int
    image: NDArray[np.uint8]


def _validate_source(source: VideoSource) -> None:
    if not source.enabled:
        raise ValueError("Video source is disabled")
    if source.source_type not in (SourceType.FILE, SourceType.RTSP):
        raise ValueError(f"Unsupported source type: {source.source_type}")


def _open_source(source: VideoSource) -> InputContainer:
    """Open file input or RTSP over TCP"""
    options = {"rtsp_transport": "tcp"} if source.source_type == SourceType.RTSP else {}
    timeout = RTSP_TIMEOUT_SECONDS if source.source_type == SourceType.RTSP else None
    return av.open(source.location, mode="r", options=options, timeout=timeout)


def _get_presentation_time(
    frame: VideoFrame, frame_index: int, previous_time: Fraction | None
) -> Fraction:
    if frame.is_corrupt:
        raise ValueError(f"Corrupt video frame at index {frame_index}")
    if frame.pts is None or not frame.time_base or frame.time_base <= 0:
        raise ValueError(f"Missing or invalid timing for frame {frame_index}")
    presentation_time = Fraction(frame.pts * frame.time_base)
    if previous_time is not None and presentation_time < previous_time:
        raise ValueError(f"Decreasing presentation time at frame {frame_index}")
    return presentation_time


def _pace_file_frame(
    source: VideoSource, elapsed: Fraction, anchor_monotonic: float, stop_event: Event
) -> bool:
    """Slow down file reading to match the video's original playback speed.

    Wait until this frame's playback time so the file behaves like a live camera.
    This avoids flooding the queue and dropping frames just because a local
    file can be read faster than real time.

    Args:
        source           : Video input; only local files need this playback delay
        elapsed          : Seconds between the first frame and this frame in the video
        anchor_monotonic : Clock reading when the first frame arrived, used to
                           measure how much real time has passed since playback started
        stop_event       : Shutdown signal that lets the wait end early

    Returns: True if shutdown interrupts the wait, otherwise False
    """
    if source.source_type == SourceType.FILE:
        delay = anchor_monotonic + float(elapsed) - time.monotonic()
        return stop_event.wait(max(0.0, delay))
    return False


def _enqueue_frames(
    video: InputContainer,
    source: VideoSource,
    frame_queue: DropOldestQueue[StreamFrame],
    selector: FrameSelector,
    stop_event: Event,
) -> None:
    """Decode and pace input, converting and enqueueing only selected frames"""
    if not video.streams.video:
        raise ValueError("Source has no video stream")
    first_time: Fraction | None = None
    previous_time: Fraction | None = None
    anchor_utc_ms = 0
    anchor_monotonic = 0.0
    for frame_index, frame in enumerate(video.decode(video.streams.video[0])):
        if stop_event.is_set():
            return
        presentation_time = _get_presentation_time(frame, frame_index, previous_time)
        previous_time = presentation_time
        if first_time is None:
            first_time = presentation_time
            anchor_utc_ms = time.time_ns() // 1_000_000
            anchor_monotonic = time.monotonic()

        elapsed = presentation_time - first_time
        if _pace_file_frame(source, elapsed, anchor_monotonic, stop_event):
            return

        media_timestamp_ms = round(presentation_time * 1000)
        if selector.should_select(media_timestamp_ms):
            selected = StreamFrame(
                source_id=source.source_id,
                frame_index=frame_index,
                media_timestamp_ms=media_timestamp_ms,
                timestamp_ms=anchor_utc_ms + round(elapsed * 1000),
                image=cast(NDArray[np.uint8], frame.to_ndarray(format="bgr24")),
            )
            frame_queue.put_nowait(selected)

    if not stop_event.is_set():
        if source.source_type == SourceType.RTSP:
            raise RuntimeError("RTSP stream ended unexpectedly")
        if first_time is None:
            raise ValueError("Source contains no decoded frames")


def run_streaming_worker(
    source: VideoSource,
    frame_queue: DropOldestQueue[StreamFrame],
    *,
    target_fps: float = 2,
    stop_event: Event,
    producer_done: Event,
) -> None:
    """Decode, select and enqueue until file EOF or a stop request.

    Args:
        source        : Enabled file or RTSP source to decode independently of recording
        frame_queue   : Bounded queue; drops its oldest waiting frame when full
        target_fps    : Maximum selected frames per second of video; defaults to 2
        stop_event    : Shared signal requesting input to stop
        producer_done : Completion signal set on exit, including on failure

    Raises:
        ValueError   : If the source or selection rate is invalid, the source has
                       no video or decoded frames, or frame data or timing is invalid
        RuntimeError : If RTSP input ends without a stop request
        FFmpegError  : If opening or decoding fails, including RTSP
                       timeouts, without a pending stop request
    """
    try:
        selector = FrameSelector(target_fps=target_fps)
        _validate_source(source)
        if stop_event.is_set():
            return
        with _open_source(source) as video:
            _enqueue_frames(video, source, frame_queue, selector, stop_event)
    except FFmpegError:
        # A pending network read may time out while responding to shutdown.
        if not stop_event.is_set():
            raise
    finally:
        producer_done.set()
        logger.debug("Streaming worker stopped | %s", source.source_id)
