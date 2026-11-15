import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from threading import Event
from time import sleep

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
MAX_CAMERA_COUNT = 4


@dataclass
class _CameraState:
    """Input and queued frames belonging to one camera

    Attributes:
        source        : Camera or file being recorded and decoded
        frames        : This camera's bounded queue of waiting frames
        producer_done : Set when this camera will produce no more frames
    """

    source: VideoSource
    frames: DropOldestQueue[StreamFrame]
    producer_done: Event = field(default_factory=Event)


@dataclass
class _PipelineState:
    """Shared state for all cameras in one run_pipeline() call

    One instance manages the whole run, whether it has one camera or four.
    Each camera has its own _CameraState with a separate frame queue. This
    shared state coordinates their workers, shutdown and errors, and tracks
    which camera the single consumer should check next.

    Attributes:
        cameras           : One _CameraState per camera, in round-robin order
        stop_event        : Signal telling every camera's workers to stop
        workers           : Futures used to wait for recording and streaming
                            workers across all cameras and check their failures
        handled_workers   : Finished workers whose results have already been
                            checked, so their errors are not reported twice
        first_error       : First failure to raise after all workers finish
                            cleanup, or None if no failure has been seen
        next_camera_index : Position in cameras to check first for the next frame
    """

    cameras: list[_CameraState]
    stop_event: Event = field(default_factory=Event)
    workers: list[Future[None]] = field(default_factory=list)
    handled_workers: set[Future[None]] = field(default_factory=set)
    first_error: BaseException | None = None
    next_camera_index: int = 0


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


def _create_pipeline_state(
    sources: Sequence[VideoSource],
    *,
    target_fps: float,
    queue_size: int,
    segment_seconds: int,
    retention_hours: float,
) -> _PipelineState:
    sources = tuple(sources)
    if not 1 <= len(sources) <= MAX_CAMERA_COUNT:
        raise ValueError(f"Provide between 1 and {MAX_CAMERA_COUNT} video sources")
    if any(not source.source_id for source in sources):
        raise ValueError("Video source IDs must not be empty")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Video source IDs must be unique")
    for source in sources:
        _validate_settings(
            source,
            target_fps=target_fps,
            segment_seconds=segment_seconds,
            retention_hours=retention_hours,
        )
    return _PipelineState(
        cameras=[
            _CameraState(source, DropOldestQueue(maxsize=queue_size))
            for source in sources
        ]
    )


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
        A worker error stops all input and is raised after the queues drain and
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
    storage_root: Path,
    *,
    session_factory: sessionmaker[Session],
    target_fps: float,
    segment_seconds: int,
    retention_hours: float,
) -> None:
    """Start recording and streaming for every camera, tracking submissions.

    Raises:
        RuntimeError: If the executor cannot accept a worker submission
    """
    for index, camera in enumerate(state.cameras):
        try:
            state.workers.append(
                executor.submit(
                    run_recording_worker,
                    camera.source,
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
                    camera.source,
                    camera.frames,
                    target_fps=target_fps,
                    stop_event=state.stop_event,
                    producer_done=camera.producer_done,
                )
            )
        except BaseException:
            # Streaming workers that never started cannot set producer_done.
            # Setting it here so the consumer does not wait forever for their frames.
            for unstarted_camera in state.cameras[index:]:
                unstarted_camera.producer_done.set()
            raise


def _get_next_frame(
    state: _PipelineState,
) -> tuple[_CameraState, StreamFrame] | None:
    """Take one FIFO frame, giving each camera a turn (round-robin)."""
    for _ in state.cameras:
        camera = state.cameras[state.next_camera_index]
        state.next_camera_index = (state.next_camera_index + 1) % len(state.cameras)
        try:
            return camera, camera.frames.get_nowait()
        except Empty:
            continue
    return None


def _is_pipeline_finished(state: _PipelineState) -> bool:
    """Wait for all workers to finish and every queue to drain.

    Check queues last so a producer's final frame is not missed.
    """
    return (
        all(camera.producer_done.is_set() for camera in state.cameras)
        and all(worker.done() for worker in state.workers)
        and all(camera.frames.empty() for camera in state.cameras)
    )


def _consume_frames(
    state: _PipelineState, handler: Callable[[StreamFrame], None]
) -> None:
    """Feed one serial consumer in round-robin order across camera queues."""
    pending: tuple[_CameraState, StreamFrame] | None = None
    while True:
        try:
            _check_workers(state)
            if pending is None:
                pending = _get_next_frame(state)
                if pending is None:
                    if _is_pipeline_finished(state):
                        return
                    sleep(FRAME_QUEUE_TIMEOUT_SECONDS)
                    continue
            camera, frame = pending
            handler(frame)
            camera.frames.task_done()
            pending = None
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
    sources: Sequence[VideoSource],
    storage_root: Path,
    *,
    session_factory: sessionmaker[Session],
    frame_handler: Callable[[StreamFrame], None],
    target_fps: float = 2,
    queue_size: int = DEFAULT_FRAME_QUEUE_SIZE,
    segment_seconds: int = 5,
    retention_hours: float = 24,
) -> None:
    """Record cameras independently and feed one shared consumer in round-robin order.

    Args:
        sources         : One to four enabled file or RTSP sources with unique source IDs.
                          Their order determines the order in which queues are checked.
        storage_root    : Base directory for recording sessions.
        session_factory : Shared factory for recording metadata transactions.
        target_fps      : Maximum selected frames per second per camera; defaults to 2.
        queue_size      : Maximum waiting frames per camera; defaults to 8. Overflow
                          drops that camera's oldest waiting frame, leaving the active frame alone.
        segment_seconds : Target recording segment duration; defaults to 5.
        retention_hours : Segment retention period in hours; defaults to 24.
        frame_handler   : One synchronous callback, called serially in the caller thread.
                          Frames carry their camera's source_id. Handles frame processing,
                          such as logging and future YOLO inference.

    Raises:
        ValueError        : If the source count, IDs, source settings or numeric settings
                            are invalid. All inputs are validated before any workers start.
        FileNotFoundError : If a file source does not refer to a regular file.

    Note:
        Each camera owns one recorder, one streaming decoder and one bounded queue.
    """
    state = _create_pipeline_state(
        sources,
        target_fps=target_fps,
        queue_size=queue_size,
        segment_seconds=segment_seconds,
        retention_hours=retention_hours,
    )
    logger.info("Pipeline started | cameras=%s", len(state.cameras))
    for camera in state.cameras:
        logger.info(
            "Camera | %s | source=%s", camera.source.source_id, camera.source.location
        )
        logger.info("Recordings | %s", storage_root / camera.source.source_id)
    logger.info(
        "Settings | target_fps=%g | segment_target=%gs | queue_size=%s",
        target_fps,
        segment_seconds,
        queue_size,
    )
    logger.info(
        "Consuming camera queues in round-robin order. Press Ctrl+C to stop and drain."
    )

    with ThreadPoolExecutor(
        max_workers=2 * len(state.cameras), thread_name_prefix="camera"
    ) as executor:
        try:
            try:
                _start_workers(
                    executor,
                    state,
                    storage_root,
                    session_factory=session_factory,
                    target_fps=target_fps,
                    segment_seconds=segment_seconds,
                    retention_hours=retention_hours,
                )
            except KeyboardInterrupt:
                logger.info("Stopping workers and draining live frames...")
                state.stop_event.set()
            except BaseException as error:
                _remember_error(state, error)
            _consume_frames(state, frame_handler)
        except BaseException as error:
            _remember_error(state, error)
        finally:
            _finish_workers(state)

    for camera in state.cameras:
        logger.info(
            "Frame queue | %s | capacity=%s | dropped_frames=%s",
            camera.source.source_id,
            camera.frames.maxsize,
            camera.frames.dropped_count,
        )
    if state.first_error is not None:
        raise state.first_error
    logger.info("Pipeline stopped.")
