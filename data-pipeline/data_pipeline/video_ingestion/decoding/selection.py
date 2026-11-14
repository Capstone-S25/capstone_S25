import math
from collections.abc import Generator, Iterable
from fractions import Fraction

from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame


class FrameSelector:
    """Select timestamps at a maximum FPS"""

    def __init__(self, *, target_fps: float) -> None:
        """
        Args:
            target_fps: Positive, finite maximum frame selection rate.
        """
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be positive and finite")
        self._interval_ms = Fraction(1000) / Fraction(str(target_fps))
        self._previous_timestamp: int | None = None
        self._last_selected_timestamp: int | None = None

    def should_select(self, timestamp_ms: int) -> bool:
        """Check whether a timestamp is far enough from the last selected frame.

        Args:
            timestamp_ms: Current frame timestamp in milliseconds, in
                nondecreasing order across calls.

        Returns:
            True for the first timestamp and each timestamp at least one
            selection interval after the last selected timestamp.
        """
        if (
            self._previous_timestamp is not None
            and timestamp_ms < self._previous_timestamp
        ):
            raise ValueError("Frame timestamps must be nondecreasing")
        self._previous_timestamp = timestamp_ms
        if (
            self._last_selected_timestamp is None
            or timestamp_ms - self._last_selected_timestamp >= self._interval_ms
        ):
            self._last_selected_timestamp = timestamp_ms
            return True
        return False


def select_frames(
    frames: Iterable[DecodedFrame], *, target_fps: float
) -> Generator[DecodedFrame, None, None]:
    """Yield frames spaced by media timestamps

    Args:
        frames: Frames from one segment, ordered by timestamp_ms.
        target_fps: Positive, finite maximum selection rate. At 2 FPS, selected
            frames must be at least 500 milliseconds apart on the media timeline.

    Yields:
        The first frame, then the first available frame at least one interval
        after the last selected frame.
    """
    selector = FrameSelector(target_fps=target_fps)
    for frame in frames:
        if selector.should_select(frame.timestamp_ms):
            yield frame
