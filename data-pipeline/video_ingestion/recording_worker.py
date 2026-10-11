from pathlib import Path
from uuid import uuid4

from video_ingestion.sources import SourceType, VideoSource


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
        "-reset_timestamps" , "1",                                 # Restart each segment’s timestamps near zero
        "-segment_list"     , str(session_dir / "completed.csv"),  # Write a list containing segment filenames and timing
        "-segment_list_type", "csv",                               
        str(session_dir / "%06d.mp4.part"),                        # Number files as 000000.mp4.part, etc
    ]  # fmt: skip

    return command
