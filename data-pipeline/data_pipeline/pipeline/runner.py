import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import perf_counter

from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.pipeline.processing import process_segment
from data_pipeline.video_ingestion.recording.worker import run_recording_worker
from data_pipeline.video_ingestion.scheduling.worker import SegmentScheduler
from data_pipeline.video_ingestion.sources import VideoSource

logger = logging.getLogger(__name__)


def run_pipeline(
    source: VideoSource,
    storage_root: Path,
    *,
    session_factory: sessionmaker[Session],
    target_fps: float = 2,
    segment_seconds: int = 5,
    retention_hours: float = 24,
) -> None:
    """Record video source in a background thread while processing segments in the main thread.

    Args:
        source: File or RTSP source to record.
        storage_root: Base directory for recording sessions.
        session_factory: Creates a separate database session for each transaction.
        target_fps: Maximum frames selected per second of video; defaults to 2.
        segment_seconds: Target recording segment duration; defaults to 5.
        retention_hours: Registration retention deadline in hours; defaults to 24.
    """
    logger.info("Pipeline started | %s | source=%s", source.source_id, source.location)
    logger.info(
        "Settings | target_fps=%g | segment_target=%gs", target_fps, segment_seconds
    )
    logger.info("Recordings | %s", storage_root / source.source_id)
    logger.info("Waiting for completed segments. Press Ctrl+C to stop.")
    scheduler = SegmentScheduler(
        source_ids=[source.source_id], session_factory=session_factory
    )
    stop_event = Event()

    # Record in the background so new video keeps arriving during processing.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="recorder") as executor:
        recording = executor.submit(
            run_recording_worker,
            source,
            storage_root,
            segment_seconds=segment_seconds,
            session_factory=session_factory,
            retention_hours=retention_hours,
            stop_event=stop_event,
        )
        try:
            while True:
                recording_finished = recording.done()
                if recording_finished:
                    # Raise any error from the recording thread.
                    recording.result()

                job = scheduler.get_next_job()
                if job is not None:
                    logger.debug(
                        "Scheduled | %s | session=%s | segment=%s | file=%s",
                        job.source_id,
                        job.session_id.hex[:8],
                        job.segment_id,
                        Path(job.path).name,
                    )
                    started = perf_counter()
                    frame_count = process_segment(
                        job,
                        storage_root,
                        session_factory,
                        target_fps=target_fps,
                    )
                    processing_seconds = perf_counter() - started
                    logger.info(
                        "Done | %s | segment=%s | file=%s | "
                        "session=%s | segment_length=%.2fs | selected=%s | processing=%.2fs",
                        job.source_id,
                        job.segment_id,
                        Path(job.path).name,
                        job.session_id.hex[:8],
                        # CSV boundaries describe a timeline interval, which can
                        # include an initial PTS offset rather than actual footage.
                        (job.end_at_ms - job.start_at_ms) / 1000,
                        frame_count,
                        processing_seconds,
                    )
                elif recording_finished:
                    # Recording finished before this query, and no pending jobs remain.
                    break
                else:
                    stop_event.wait(0.25)
        except KeyboardInterrupt:
            logger.info("Stopping recording, scheduling, and decoding...")
        finally:
            # Ask the recorder to stop, whether processing finished or raised an error.
            stop_event.set()
            # Wait for the recorder to save its final segment metadata before
            # the caller closes the database engine.
            recording.result()

    logger.info("Pipeline stopped.")
