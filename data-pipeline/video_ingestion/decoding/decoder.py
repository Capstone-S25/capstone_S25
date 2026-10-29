from collections.abc import Generator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast
from uuid import UUID

# Using PyAV beucase it uses FFmpeg consistently;
# OpenCV's video backend can vary by platform and build.
import av
import numpy as np
from numpy.typing import NDArray

from video_ingestion.scheduling.worker import SegmentJob


@dataclass(frozen=True)
class DecodedFrame:
    """One original-resolution image and its recording metadata.

    Attributes:
        segment_id: Database ID of the segment containing this frame.
        source_id: Camera or file source identifier.
        session_id: Recording session UUID.
        frame_index: Zero-based decoded frame index within the segment.
        timestamp_ms: Approximate UTC epoch milliseconds, based on the segment
            start and segment-local presentation time, including its initial
            offset; not a camera capture timestamp.
        image: BGR uint8 pixel array with shape (height, width, 3).
    """

    segment_id: int
    source_id: str
    session_id: UUID
    frame_index: int
    timestamp_ms: int
    image: NDArray[np.uint8]


def decode_segment(
    job: SegmentJob, storage_root: Path
) -> Generator[DecodedFrame, None, None]:
    """Yield every frame of a recorded segment at its original resolution.

    Args:
        job: Claimed segment metadata, including its relative file path and
            approximate UTC start time.
        storage_root: Recordings directory used to resolve job.path. Resolved
            paths must stay inside this directory, including symlink targets.

    Yields:
        DecodedFrame objects from the first video stream, in presentation order.
        Images are BGR uint8 arrays; frame indices start at zero. Timestamps
        add each frame's segment-local presentation time to job.start_at_ms.
        Initial timestamp offsets are preserved, not subtracted. This expects
        segments recorded with FFmpeg's reset_timestamps=1 and job.start_at_ms
        derived from the corresponding segment-list CSV start time.

    Raises:
        ValueError: If the path is absolute, escapes storage_root, is not a
            regular file, or the file has no video stream or decoded frames. Also raised for
            corrupt frames, missing/invalid timing, or decreasing timestamps.
        OSError: If the directory or file is missing or cannot be accessed.
        av.error.FFmpegError: If opening or decoding the video fails.

    Note:
        Opening and validation happen when iteration starts. The container
        closes on exhaustion, decoding errors, or generator.close(). A caller
        that may stop early must explicitly close the generator, for example:

        with contextlib.closing(decode_segment(job, storage_root)) as frames:
            for frame in frames:
                break
    """
    relative_path = Path(job.path)
    if relative_path.is_absolute() or relative_path.anchor:
        raise ValueError("Segment path must be relative to the recordings root")

    root = storage_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Recordings root is not a directory: {root}")

    segment_path = (root / relative_path).resolve(strict=True)
    if not segment_path.is_relative_to(root):
        raise ValueError("Segment path resolves outside the recordings root")
    if not segment_path.is_file():
        raise ValueError(f"Segment path is not a regular file: {segment_path}")

    with av.open(str(segment_path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"Segment has no video stream: {segment_path}")
        stream = container.streams.video[0]
        previous_time: Fraction | None = None
        decoded_any = False

        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.is_corrupt:
                raise ValueError(f"Corrupt video frame at index {frame_index}")
            if frame.pts is None or not frame.time_base or frame.time_base <= 0:
                raise ValueError(f"Missing or invalid timing for frame {frame_index}")

            # Rational arithmetic to avoid FPS guesses and floating-point rounding drift
            # pts: the frame's presentation timestamp in ticks
            # time_base: seconds per tick.
            # Multiplying them gives presentation time in seconds
            presentation_time = Fraction(frame.pts * frame.time_base)
            if previous_time is not None and presentation_time < previous_time:
                raise ValueError(
                    f"Decreasing presentation timestamp at frame {frame_index}"
                )
            previous_time = presentation_time
            # Preserve the initial PTS offset so adjacent segments keep their
            # positions on the recording timeline.
            presentation_ms = round(presentation_time * 1000)

            decoded_any = True
            yield DecodedFrame(
                segment_id=job.segment_id,
                source_id=job.source_id,
                session_id=job.session_id,
                frame_index=frame_index,
                # Summary for timestamp_ms:
                # FFmpeg startup reference (UTC)
                #     + CSV segment start offset
                #     = job.start_at_ms
                #         + frame PTS × time_base
                #         = estimated frame UTC timestamp
                timestamp_ms=job.start_at_ms + presentation_ms,
                # bgr24 guarantees uint8; PyAV's general return type also lists
                # dtypes used by other pixel formats
                image=cast(NDArray[np.uint8], frame.to_ndarray(format="bgr24")),
            )

        if not decoded_any:
            raise ValueError(f"Segment contains no decoded frames: {segment_path}")
