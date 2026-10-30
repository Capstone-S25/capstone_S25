import math
from collections.abc import Generator, Iterable
from fractions import Fraction

from video_ingestion.decoding.decoder import DecodedFrame


def select_frames(
    frames: Iterable[DecodedFrame], *, target_fps: float
) -> Generator[DecodedFrame, None, None]:
    """Yield frames spaced by media timestamps, independent of processing speed.

    Args:
        frames: Frames from one segment, ordered by timestamp_ms.
        target_fps: Positive, finite maximum selection rate. At 2 FPS, selected
            frames must be at least 500 milliseconds apart on the media timeline.

    Yields:
        The first frame, then the first available frame at least one interval
        after the last selected frame.

    Raises:
        ValueError: If target_fps is nonpositive/nonfinite or input timestamps
            decrease, including on frames that would otherwise be skipped.
    """
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite")

    # Keep fractional rates precise without converting epoch timestamps to floats.
    interval_ms = Fraction(1000) / Fraction(str(target_fps))
    previous_timestamp: int | None = None
    last_selected_timestamp: int | None = None

    for frame in frames:
        timestamp = frame.timestamp_ms
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("Frame timestamps must be nondecreasing")
        previous_timestamp = timestamp

        if (
            last_selected_timestamp is None
            or timestamp - last_selected_timestamp >= interval_ms
        ):
            last_selected_timestamp = timestamp
            yield frame
