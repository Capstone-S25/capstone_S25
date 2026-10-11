import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from video_ingestion.sources import SourceType, VideoSource

_SEGMENT_LIST_FILENAME = "completed.csv"


@dataclass(frozen=True)
class SegmentEntry:
    """One row from FFmpeg's segment-list CSV.

    filename is the segment name as written in the CSV.
    start_seconds and end_seconds mark its interval on the recording timeline.
    """

    filename: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class FinalizedSegment:
    path: Path
    start_seconds: float
    end_seconds: float
    duration_seconds: float


def _prepare_session(
    source: VideoSource,
    storage_root: Path,
) -> tuple[str, Path]:
    """Prepare an empty folder for one recording session;

    Args:
        source       : Video source whose source_id names the parent folder.
        storage_root : Base directory for recordings, e.g. Path("recordings").

    Returns:
        A (session_id, session_dir) tuple. The ID is a new UUID4 hex string,
        and session_dir is storage_root / source.source_id / session_id.
        For "camera_01", this creates recordings/camera_01/<session_id>/,
        including any missing parent directories.

    Raises:
        ValueError : If source.enabled is False; no directory is created.
        OSError    : If the directory cannot be created, e.g. permission denied.
    """

    if not source.enabled:
        raise ValueError("Video source is disabled")

    session_id = uuid4().hex

    session_dir = storage_root / source.source_id / session_id

    # Create folder and any missing parent folders
    session_dir.mkdir(parents=True, exist_ok=True)

    return session_id, session_dir


def _build_input_args(source: VideoSource) -> list[str]:
    """Build FFmpeg input arguments according to the given video source.

    Args:
        source: Video source providing the input type and file path or RTSP URL.

    Returns:
        Arguments to include in an FFmpeg command.

    Raises:
        ValueError: If the source type is unsupported.
    """

    if source.source_type == SourceType.FILE:
        return [
            "-re",  # Read the file at its original speed, simulating a live stream
            "-i",
            source.location,
        ]

    if source.source_type == SourceType.RTSP:
        return ["-rtsp_transport", "tcp", "-i", source.location]

    raise ValueError(f"Unsupported source type: {source.source_type}")


def _build_ffmpeg_command(
    source: VideoSource, session_dir: Path, segment_seconds: int = 5
) -> list[str]:
    """Build a command to copy the first video stream into MP4 segments.

    Args:
        source: File or RTSP input; audio is omitted from the output.
        session_dir: Output directory.
            The caller must create it before running the command.
        segment_seconds: Target segment duration in seconds; defaults to 5.
            Stream copying cuts at keyframes (frames that can be decoded
            without earlier frames), so actual durations can vary.

    Returns:
        Argument list for subprocess, starting with "ffmpeg". When executed,
        writes numbered files such as 000000.mp4.part and completed.csv with
        segment filenames, start times, and end times in seconds.

    Raises:
        ValueError: If segment_seconds is nonpositive or the source type is
            unsupported.
    """

    if segment_seconds <= 0:
        raise ValueError("Segment duration must be positive")

    session_dir = session_dir.resolve()

    command = ["ffmpeg", "-hide_banner"]

    command += _build_input_args(source)

    command += [
        "-map"              , "0:v:0",  # Select the first video stream from the first input
        "-c:v"              , "copy",   # Copy compressed video without decoding and re-encoding
        "-an"               ,           # Omit audio
        # Split video into MP4 segments
        "-f"                , "segment",
        "-segment_format"   , "mp4",
        "-segment_time"     , str(segment_seconds),
        # Record completed segments and set the output filename
        "-reset_timestamps" , "1",                                        # Restart each segment’s timestamps near zero
        "-segment_list"     , str(session_dir / _SEGMENT_LIST_FILENAME),  # Write a list containing segment filenames and timing
        "-segment_list_type", "csv",                               
        str(session_dir / "%06d.mp4.part"),                               # Number files as 000000.mp4.part, etc
    ]  # fmt: skip

    return command


def _start_ffmpeg(command: list[str]) -> subprocess.Popen:
    """Start FFmpeg without waiting for recording to finish.

    Args:
        command: Argument list from _build_ffmpeg_command. The FFmpeg executable
            must be available, and the output directory must already exist.

    Returns:
        A process handle for monitoring and controlling FFmpeg.

    Raises:
        OSError: If the executable cannot be launched, e.g. FFmpeg is missing.
    """

    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,  # Lets Python send commands to FFmpeg later
        text=True,  # Allows writing strings to that input pipe
    )


def _read_segment_entries(
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


def _probe_segment_duration(segment_path: Path) -> float:
    """Validate a segment with FFprobe and return its file duration.

    Args:
        segment_path: Path to the recorded segment to inspect.

    Returns:
        A finite, positive file duration in seconds.

    Raises:
        ValueError: If no video stream exists or the duration is missing,
            nonnumeric, nonfinite, or nonpositive.
        json.JSONDecodeError: If FFprobe's output is not valid JSON.
        subprocess.CalledProcessError: If FFprobe exits with a nonzero code.
        subprocess.TimeoutExpired: If FFprobe does not finish within 10 seconds.
        OSError: If FFprobe cannot be launched, e.g. the executable is missing.
    """

    command = [
        "ffprobe",
        "-v"             , "error",                         # Show only error messages (for stderr)
        "-select_streams", "v:0",
        "-show_entries"  , "stream=index:format=duration",  # Request stream index and file duration
        "-of"            , "json",                          # Output metadata as JSON
        str(segment_path),
    ]  # fmt: skip

    result = subprocess.run(
        command,
        capture_output=True,  # Capture FFprobe's output and error messages
        text=True,  # Return strings rather than bytes
        check=True,  # Raise CalledProcessError if FFprobe fails
        timeout=10,
    )

    metadata = json.loads(result.stdout)

    if not metadata.get("streams"):
        raise ValueError(f"No video stream found: {segment_path}")

    duration = float(metadata.get("format", {}).get("duration", "nan"))

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid segment duration: {segment_path}")

    return duration


def _finalize_segment(
    entry: SegmentEntry,
    session_dir: Path,
) -> FinalizedSegment:
    """Validate and rename one temporary segment, then return its metadata.

    Args:
        entry: Completed CSV entry containing the temporary filename and its
            start/end times on the recording timeline.
        session_dir: Directory containing the recorded segment.

    Returns:
        A FinalizedSegment with the renamed file path, the entry's start/end
        times, and the duration reported by FFprobe in seconds.
        File rename example: 000000.mp4.part becomes 000000.mp4.

    Raises:
        ValueError: If the filename does not end in .part, no video stream
            exists, or the probed duration is missing or invalid.
        FileExistsError: If the finalized destination already exists.
        json.JSONDecodeError: If FFprobe's output is not valid JSON.
        subprocess.CalledProcessError: If FFprobe exits with a nonzero code.
        subprocess.TimeoutExpired: If FFprobe does not finish within 10 seconds.
        OSError: If FFprobe cannot be launched or the file cannot be renamed,
            e.g. the temporary file is missing or permission is denied.
    """
    part_path = session_dir / entry.filename

    if part_path.suffix != ".part":
        raise ValueError(f"Expected a temporary segment: {part_path}")

    # Remove only ".part": 000000.mp4.part becomes 000000.mp4
    final_path = part_path.with_suffix("")

    if final_path.exists():
        raise FileExistsError(f"Final segment already exists: {final_path}")

    duration = _probe_segment_duration(part_path)

    part_path.rename(final_path)

    return FinalizedSegment(
        path=final_path,
        start_seconds=entry.start_seconds,
        end_seconds=entry.end_seconds,
        duration_seconds=duration,
    )


def run_recording_worker(
    source: VideoSource,
    storage_root: Path,
    segment_seconds: int = 5,
):
    # Create a recording session and its output directory
    session_id, session_dir = _prepare_session(source, storage_root)

    # Build the command and launch FFmpeg in the background
    command = _build_ffmpeg_command(source, session_dir, segment_seconds)
    process = _start_ffmpeg(command)

    stopped_by_user = False
    csv_path = session_dir / _SEGMENT_LIST_FILENAME
    csv_offset = 0

    try:
        # poll() returns None while FFmpeg is still running
        while process.poll() is None:
            entries, csv_offset = _read_segment_entries(csv_path, csv_offset)

            for entry in entries:
                # TODO: produce the video segment metadata
                print(entry, flush=True)
                duration = _probe_segment_duration(session_dir / entry.filename)
                print(duration, flush=True)

            time.sleep(0.25)

    except KeyboardInterrupt:
        stopped_by_user = True

        if process.poll() is None:
            try:
                process.communicate(input="q\n", timeout=10)
            except subprocess.TimeoutExpired:
                # Force FFmpeg to stop if it does not respond in time
                process.kill()
                process.wait()
                raise RuntimeError("FFmpeg did not stop within 10 seconds") from None
    finally:
        if process.poll() is None:
            process.kill()

        # Wait for process termination
        process.wait()

        # Release Python's end of the input pipe
        if process.stdin is not None:
            process.stdin.close()

    # Edge case: FFmpeg may write the final rows between the last poll and process exit
    entries, csv_offset = _read_segment_entries(csv_path, csv_offset)
    for entry in entries:
        # TODO: produce the video segment metadata
        print(entry, flush=True)
        duration = _probe_segment_duration(session_dir / entry.filename)
        print(duration, flush=True)

    if process.returncode != 0 and not stopped_by_user:
        raise RuntimeError(f"FFmpeg exited with code {process.returncode}")
