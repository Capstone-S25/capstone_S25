import logging
import math
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.database.repositories.video_segments import (
    insert_video_segment_metadata,
)
from data_pipeline.video_ingestion.recording.ffmpeg import (
    SEGMENT_LIST_FILENAME,
    build_segment_recording_command,
    start_ffmpeg,
)
from data_pipeline.video_ingestion.recording.segments import (
    FinalizedSegment,
    SegmentEntry,
    finalize_segment_with_retry,
    read_segment_entries,
)
from data_pipeline.video_ingestion.sources import VideoSource

logger = logging.getLogger(__name__)


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


def _register_segment_metadata_with_retry(
    segment: FinalizedSegment,
    *,
    source_id: str,
    session_id: str,
    storage_root: Path,
    session_factory: sessionmaker[Session],
    retention_hours: float,
    max_attempts: int = 3,
) -> None:
    """Register an already finalized file, retrying database operational errors.

    Each attempt uses a fresh transaction, including commit in the retry scope.
    The retention deadline stays fixed across attempts. After max_attempts,
    propagate the error and leave the file intact for later recovery. Integrity
    and validation errors are not retried; duplicate paths remain a no-op.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    delete_at = datetime.now(UTC) + timedelta(hours=retention_hours)
    for attempt in range(max_attempts):
        try:
            with session_factory.begin() as db:
                insert_video_segment_metadata(
                    db,
                    source_id=source_id,
                    session_id=session_id,
                    storage_root=storage_root,
                    path=segment.path,
                    start_at_ms=segment.start_at_ms,
                    end_at_ms=segment.end_at_ms,
                    delete_at=delete_at,
                )
            return
        except (OperationalError, InterfaceError):
            if attempt == max_attempts - 1:
                raise
            time.sleep(0.25)


def _finalize_and_register_segment(
    entry: SegmentEntry,
    session_dir: Path,
    *,
    source: VideoSource,
    session_id: str,
    storage_root: Path,
    session_started_at_utc: int,
    session_factory: sessionmaker[Session],
    retention_hours: float,
) -> FinalizedSegment:
    segment = finalize_segment_with_retry(
        entry, session_dir, session_started_at_utc=session_started_at_utc
    )
    _register_segment_metadata_with_retry(
        segment,
        source_id=source.source_id,
        session_id=session_id,
        storage_root=storage_root,
        session_factory=session_factory,
        retention_hours=retention_hours,
    )
    return segment


def _stop_ffmpeg(process: subprocess.Popen) -> None:
    """Ask FFmpeg to close the current segment, forcing termination on timeout."""
    if process.poll() is None:
        try:
            process.communicate(input="q\n", timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("FFmpeg did not stop within 10 seconds") from None


def run_recording_worker(
    source: VideoSource,
    storage_root: Path,
    segment_seconds: int = 5,
    *,
    session_factory: sessionmaker[Session],
    retention_hours: float = 24,
    stop_event: Event | None = None,
):
    """Record and register segments until EOF, Ctrl+C, or a stop request.

    Args:
        source: Enabled file or RTSP video source to record.
        storage_root: Base directory for recordings. Each run creates a new
            session under storage_root / source.source_id / session_id.
        segment_seconds: Target segment duration in seconds; defaults to 5.
            Cuts occur at keyframes, so actual segment lengths can vary.
        session_factory: Factory for short-lived transactions that register
            finalized segment metadata and commit each insertion.
        retention_hours: Positive, finite hours until each registered segment's
            retention deadline; defaults to 24.
        stop_event: Optional shared shutdown signal for a background recorder.
            Set it to request graceful shutdown. Defaults to None.

    Raises:
        ValueError: If the source is disabled or unsupported, segment_seconds
            is nonpositive, retention_hours is not positive and finite, or a
            completed CSV entry or segment filename is invalid.
        OSError: If FFmpeg cannot be launched or recording files cannot be
            created, read, or renamed.
        csv.Error: If a completed segment-list row contains malformed CSV.
        sqlalchemy.exc.SQLAlchemyError: If metadata registration fails after
            applicable retries. Already finalized video files remain on disk.
        RuntimeError: If FFmpeg exits unsuccessfully without a stop request,
            or does not stop within 10 seconds of a graceful shutdown request.
    """
    if not math.isfinite(retention_hours) or retention_hours <= 0:
        raise ValueError("retention_hours must be positive and finite")

    # Create a recording session and its output directory
    session_id, session_dir = _prepare_session(source, storage_root)

    # Build the command and launch FFmpeg in the background
    command = build_segment_recording_command(source, session_dir, segment_seconds)
    # TODO: Map video timestamps to the CCTV camera's synchronized UTC clock
    # for more accurate capture times; FFmpeg startup time is an approximation.
    session_started_at_utc = time.time_ns() // 1_000_000
    process = start_ffmpeg(command)

    stopped_by_user = False
    csv_path = session_dir / SEGMENT_LIST_FILENAME
    csv_offset = 0

    try:
        # poll() returns None while FFmpeg is still running
        while process.poll() is None:
            if stop_event is not None and stop_event.is_set():
                stopped_by_user = True
                _stop_ffmpeg(process)
                break

            entries, csv_offset = read_segment_entries(csv_path, csv_offset)

            for entry in entries:
                segment = _finalize_and_register_segment(
                    entry,
                    session_dir,
                    source=source,
                    session_id=session_id,
                    storage_root=storage_root,
                    session_started_at_utc=session_started_at_utc,
                    session_factory=session_factory,
                    retention_hours=retention_hours,
                )

                logger.debug(
                    "Recorded | %s | session=%s | file=%s",
                    source.source_id,
                    session_id[:8],
                    segment.path.name,
                )

            time.sleep(0.25)

    except KeyboardInterrupt:
        stopped_by_user = True

        _stop_ffmpeg(process)
    finally:
        if process.poll() is None:
            process.kill()

        # Wait for process termination
        process.wait()

        # Release Python's end of the input pipe
        if process.stdin is not None:
            process.stdin.close()

    # Edge case: FFmpeg may write the final rows between the last poll and process exit
    entries, csv_offset = read_segment_entries(csv_path, csv_offset)
    for entry in entries:
        segment = _finalize_and_register_segment(
            entry,
            session_dir,
            source=source,
            session_id=session_id,
            storage_root=storage_root,
            session_started_at_utc=session_started_at_utc,
            session_factory=session_factory,
            retention_hours=retention_hours,
        )

        logger.debug(
            "Recorded | %s | session=%s | file=%s",
            source.source_id,
            session_id[:8],
            segment.path.name,
        )

    if process.returncode != 0 and not stopped_by_user:
        raise RuntimeError(f"FFmpeg exited with code {process.returncode}")
