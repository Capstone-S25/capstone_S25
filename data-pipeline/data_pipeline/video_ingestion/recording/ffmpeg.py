import subprocess
from pathlib import Path

from data_pipeline.video_ingestion.sources import SourceType, VideoSource

SEGMENT_LIST_FILENAME = "completed.csv"


def _build_recording_input_args(source: VideoSource) -> list[str]:
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


def build_segment_recording_command(
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

    # Show warnings/errors without the banner or per-frame progress counters.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats"]

    command += _build_recording_input_args(source)

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
        "-segment_list"     , str(session_dir / SEGMENT_LIST_FILENAME),  # Write a list containing segment filenames and timing
        "-segment_list_type", "csv",                               
        str(session_dir / "%06d.mp4.part"),                               # Number files as 000000.mp4.part, etc
    ]  # fmt: skip

    return command


def start_ffmpeg(command: list[str]) -> subprocess.Popen:
    """Start FFmpeg without waiting for recording to finish.

    Args:
        command: Argument list from build_segment_recording_command. The FFmpeg executable
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
