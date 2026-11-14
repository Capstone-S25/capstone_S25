import logging
import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from threading import Event

from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.video_ingestion.recording.worker import run_recording_worker
from data_pipeline.video_ingestion.sources import SourceType, VideoSource
from data_pipeline.video_ingestion.streaming.frame_queue import (
    DEFAULT_FRAME_QUEUE_SIZE,
    DropOldestQueue,
)
from data_pipeline.video_ingestion.streaming.worker import (
    StreamFrame,
    run_streaming_worker,
)

logger = logging.getLogger(__name__)

FRAME_QUEUE_TIMEOUT_SECONDS = 0.1


@dataclass
class _PipelineState:
    """Resources and failure state owned by one pipeline invocation

    Attributes:
        stop_event      : Shared signal requesting both workers to stop
        producer_done   : Set when streaming finishes, including on failure
        frames          : Bounded frame queue; drops the oldest waiting frame when full
        workers         : Futures for successfully submitted recording and streaming workers
        handled_workers : Finished workers already checked, so the same error
                          is not reported again
        first_error     : First observed failure to propagate after cleanup, or None
    """

    stop_event: Event = field(default_factory=Event)
    producer_done: Event = field(default_factory=Event)
    frames: DropOldestQueue[StreamFrame] = field(default_factory=DropOldestQueue)
    workers: list[Future[None]] = field(default_factory=list)
    handled_workers: set[Future[None]] = field(default_factory=set)
    first_error: BaseException | None = None


def _log_frame(frame: StreamFrame) -> None:
    logger.info(
        "Live frame | %s | frame=%s | media_ms=%s | timestamp_ms=%s | shape=%s",
        frame.source_id,
        frame.frame_index,
        frame.media_timestamp_ms,
        frame.timestamp_ms,
        frame.image.shape,
    )


def _validate_settings(
    source: VideoSource,
    *,
    target_fps: float,
    segment_seconds: int,
    retention_hours: float,
) -> None:
    if not source.enabled:
        raise ValueError("Video source is disabled")
    if source.source_type not in (SourceType.FILE, SourceType.RTSP):
        raise ValueError(f"Unsupported source type: {source.source_type}")
    if not source.location:
        raise ValueError("Video source location must not be empty")
    if source.source_type == SourceType.FILE and not Path(source.location).is_file():
        raise FileNotFoundError(f"Video not found: {source.location}")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("Segment duration must be positive and finite")
    if not math.isfinite(retention_hours) or retention_hours <= 0:
        raise ValueError("retention_hours must be positive and finite")


def _remember_error(state: _PipelineState, error: BaseException) -> None:
    state.stop_event.set()
    if state.first_error is None:
        state.first_error = error
    elif error is not state.first_error:
        logger.error(
            "Additional worker or cleanup failure",
            exc_info=(type(error), error, error.__traceback__),
        )


def _check_workers(state: _PipelineState) -> None:
    """Observe completed workers without blocking frame consumption.

    Note:
        Check each finished worker once to avoid reporting the same error again.
        A worker error stops input and is raised after the queue drains and
        cleanup finishes.
    """
    for future in state.workers:
        if future.done() and future not in state.handled_workers:
            state.handled_workers.add(future)
            error = future.exception()
            if error is not None:
                _remember_error(state, error)


def _start_workers(
    executor: ThreadPoolExecutor,
    state: _PipelineState,
    source: VideoSource,
    storage_root: Path,
    *,
    session_factory: sessionmaker[Session],
    target_fps: float,
    segment_seconds: int,
    retention_hours: float,
) -> None:
    """Start recording and streaming, tracking each successful submission.

    Raises:
        RuntimeError: If the executor cannot accept a worker submission
    """
    state.workers.append(
        executor.submit(
            run_recording_worker,
            source,
            storage_root,
            segment_seconds=segment_seconds,
            session_factory=session_factory,
            retention_hours=retention_hours,
            stop_event=state.stop_event,
        )
    )
    state.workers.append(
        executor.submit(
            run_streaming_worker,
            source,
            state.frames,
            target_fps=target_fps,
            stop_event=state.stop_event,
            producer_done=state.producer_done,
        )
    )


def _consume_frames(
    state: _PipelineState, handler: Callable[[StreamFrame], None]
) -> None:
    """Consume FIFO frames, draining on Ctrl+C or worker failure.

    Note:
        Returns once both workers finish and the queue is empty Ctrl+C stops
        input and retries any interrupted callback before draining the queue
    """
    pending_frame: StreamFrame | None = None
    while True:
        try:
            _check_workers(state)
            if pending_frame is None:
                try:
                    pending_frame = state.frames.get(
                        timeout=FRAME_QUEUE_TIMEOUT_SECONDS
                    )
                except Empty:
                    if state.producer_done.is_set() and all(
                        worker.done() for worker in state.workers
                    ):
                        return
                    continue
            handler(pending_frame)
            state.frames.task_done()
            pending_frame = None
        except KeyboardInterrupt:
            logger.info("Stopping workers and draining live frames...")
            state.stop_event.set()


def _finish_workers(state: _PipelineState) -> None:
    """Wait for cleanup and final recording registration without masking errors.

    Note:
        Saves failures that have not been handled yet and logs additional errors
        without replacing the first failure. Does not re-raise worker failures
    """
    state.stop_event.set()
    for future in state.workers:
        try:
            future.result()
        except BaseException as error:
            if future not in state.handled_workers:
                state.handled_workers.add(future)
                _remember_error(state, error)


def run_pipeline(
    source: VideoSource,
    storage_root: Path,
    *,
    session_factory: sessionmaker[Session],
    target_fps: float = 2,
    queue_size: int = DEFAULT_FRAME_QUEUE_SIZE,
    segment_seconds: int = 5,
    retention_hours: float = 24,
    frame_handler: Callable[[StreamFrame], None] | None = None,
) -> None:
    """Record and decode independently, consuming live frames in the caller thread.

    Args:
        source: Enabled file or RTSP source to record and decode independently.
        storage_root: Base directory for recording sessions.
        session_factory: Factory for recording metadata transactions.
        target_fps: Maximum selected frames per second of video; defaults to 2.
        queue_size: Maximum waiting frames per camera; defaults to 8. A full
            queue drops its oldest waiting frame to accept the newest frame.
        segment_seconds: Target recording segment duration; defaults to 5.
        retention_hours: Segment retention period in hours; defaults to 24.
        frame_handler: Synchronous callback receiving each selected StreamFrame.
            Defaults to logging frame metadata when None; can later run YOLO.

    Raises:
        ValueError: If the source or numeric settings are invalid.
        FileNotFoundError: If a file source does not refer to a regular file.

    Note:
        Each invocation owns one recorder, one decoder and a bounded frame queue.
        Overflow drops waiting frames; the frame being processed is not interrupted.
        Ctrl+C stops input and drains the queue.
        Worker failures stop the peer and drain the queue.
        Callback failures stop both workers without further callback calls.
        The first observed failure is propagated after cleanup; additional failures are logged.
        All workers finish before returning so the caller can safely close its database.
    """
    _validate_settings(
        source,
        target_fps=target_fps,
        segment_seconds=segment_seconds,
        retention_hours=retention_hours,
    )
    state = _PipelineState(frames=DropOldestQueue(maxsize=queue_size))
    logger.info("Pipeline started | %s | source=%s", source.source_id, source.location)
    logger.info(
        "Settings | target_fps=%g | segment_target=%gs | queue_size=%s",
        target_fps,
        segment_seconds,
        queue_size,
    )
    logger.info("Recordings | %s", storage_root / source.source_id)
    logger.info("Consuming live frames. Press Ctrl+C to stop and drain the queue.")
    handler = frame_handler if frame_handler is not None else _log_frame

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera") as executor:
        try:
            _start_workers(
                executor,
                state,
                source,
                storage_root,
                session_factory=session_factory,
                target_fps=target_fps,
                segment_seconds=segment_seconds,
                retention_hours=retention_hours,
            )
            _consume_frames(state, handler)
        except BaseException as error:
            _remember_error(state, error)
        finally:
            _finish_workers(state)

    logger.info(
        "Frame queue | capacity=%s | dropped_frames=%s",
        state.frames.maxsize,
        state.frames.dropped_count,
    )
    if state.first_error is not None:
        raise state.first_error
    logger.info("Pipeline stopped.")
