import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentEntry:
    """One row from FFmpeg's segment-list CSV.

    Attributes:
        filename: Segment filename as written in the CSV, e.g. "000000.mp4.part".
        start_seconds: Segment start on the recording timeline, in seconds.
        end_seconds: Segment end on the recording timeline, in seconds.
    """

    filename: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class FinalizedSegment:
    """Metadata for a segment after its .part suffix is removed.

    Note:
    Times add CSV offsets to the FFmpeg startup reference; startup delay and
    buffering affect accuracy. They are not packet arrival or capture times.

    Attributes:
        path: Finalized file path with the .part suffix removed.
        start_at_ms: Approximate segment start as UTC epoch milliseconds.
        end_at_ms: Approximate segment end as UTC epoch milliseconds.
    """

    path: Path
    start_at_ms: int
    end_at_ms: int


def read_segment_entries(
    csv_path: Path,
    offset: int = 0,
) -> tuple[list[SegmentEntry], int]:
    """Read complete new rows from FFmpeg's segment-list CSV while it grows.

    Args:
        csv_path: Segment-list file, e.g. session_dir / "completed.csv".
            Each row contains a filename, start time, and end time in seconds.
        offset: Read position returned by the previous call; use 0 initially.

    Returns:
        A (entries, next_offset) tuple. Entries contain filenames as written in
        the CSV and start/end times converted to floats. Pass next_offset to
        the next call to avoid rereading rows. A missing file returns an empty
        list and the original offset. An unfinished row without a trailing
        newline is deferred until a later call, without advancing past it.

    Raises:
        ValueError: If a complete row has other than three fields or a time
            cannot be converted to float.
        csv.Error: If a complete row contains malformed CSV syntax.
        OSError: If the file cannot be read, except when it does not exist.
    """
    entries: list[SegmentEntry] = []

    try:
        file = csv_path.open("r", encoding="utf-8", newline="")
    except FileNotFoundError:
        return entries, offset

    with file:
        file.seek(offset)

        while True:
            line = file.readline()

            # Stop at EOF or a row FFmpeg has not finished writing
            if not line or not line.endswith("\n"):
                break

            row = next(csv.reader([line], strict=True))

            if len(row) != 3:
                raise ValueError(f"Invalid segment-list row: {row!r}")

            entries.append(
                SegmentEntry(
                    filename=row[0],
                    start_seconds=float(row[1]),
                    end_seconds=float(row[2]),
                )
            )

            offset = file.tell()

    return entries, offset


def _finalize_segment(
    entry: SegmentEntry,
    session_dir: Path,
    *,
    session_started_at_utc: int,
) -> FinalizedSegment:
    """Rename a completed temporary segment and return approximate UTC metadata.

    Args:
        entry: Completed CSV entry containing the temporary filename and its
            start/end times on the recording timeline.
        session_dir: Directory containing the recorded segment.
        session_started_at_utc: UTC epoch milliseconds captured before FFmpeg
            startup; reused unchanged for every segment and retry.

    Returns:
        A FinalizedSegment with the renamed path and UTC epoch milliseconds
        calculated as the session reference plus rounded CSV offsets. Only
        .part is removed: 000000.mp4.part becomes 000000.mp4.

    Raises:
        ValueError: If the filename does not end in .part or timestamps are
            nonfinite, negative, or out of order. Validation precedes renaming.
        FileExistsError: If the finalized destination already exists.
        OSError: If the file cannot be renamed, e.g. it is missing or permission
            is denied.
    """
    part_path = session_dir / entry.filename

    if part_path.suffix != ".part":
        raise ValueError(f"Expected a temporary segment: {part_path}")

    if not all(
        math.isfinite(value) for value in (entry.start_seconds, entry.end_seconds)
    ):
        raise ValueError("Segment timestamps must be finite")
    if entry.start_seconds < 0 or entry.end_seconds < entry.start_seconds:
        raise ValueError("Segment timestamps must satisfy 0 <= start <= end")

    start_offset_ms = entry.start_seconds * 1000
    end_offset_ms = entry.end_seconds * 1000
    if not all(math.isfinite(value) for value in (start_offset_ms, end_offset_ms)):
        raise ValueError("Segment timestamps are too large to convert to milliseconds")
    start_at_ms = session_started_at_utc + round(start_offset_ms)
    end_at_ms = session_started_at_utc + round(end_offset_ms)

    # Remove only ".part": 000000.mp4.part becomes 000000.mp4
    final_path = part_path.with_suffix("")

    if final_path.exists():
        raise FileExistsError(f"Final segment already exists: {final_path}")

    part_path.rename(final_path)

    return FinalizedSegment(
        path=final_path,
        start_at_ms=start_at_ms,
        end_at_ms=end_at_ms,
    )


def finalize_segment_with_retry(
    entry: SegmentEntry,
    session_dir: Path,
    max_attempts: int = 3,
    *,
    session_started_at_utc: int,
) -> FinalizedSegment:
    attempts = 0

    while True:
        try:
            return _finalize_segment(
                entry, session_dir, session_started_at_utc=session_started_at_utc
            )

        # On Windows, an open file handle that does not allow renaming can cause
        # Path.rename to raise PermissionError. A short delay lets a temporary file
        # lock clear before another attempt
        except PermissionError:
            attempts += 1

            if attempts >= max_attempts:
                raise

            time.sleep(0.25)
