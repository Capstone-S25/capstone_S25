import logging
from contextlib import closing
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from data_pipeline.database.models.video_segments import SegmentStatus, VideoSegment
from data_pipeline.database.repositories.video_segments import (
    update_video_segment_status,
)
from data_pipeline.video_ingestion.decoding.decoder import decode_segment
from data_pipeline.video_ingestion.decoding.selection import select_frames
from data_pipeline.video_ingestion.scheduling.worker import SegmentJob
from data_pipeline.video_ingestion.streaming.worker import StreamFrame

logger = logging.getLogger(__name__)


def process_frame(frame: StreamFrame) -> None:
    """Handle one live frame from the shared round-robin consumer.

    Args:
        frame : Selected frame, including its camera ID, timestamps and BGR image
    """
    logger.info(
        "Live frame | %s | frame=%s | media_ms=%s | timestamp_ms=%s | shape=%s",
        frame.source_id,
        frame.frame_index,
        frame.media_timestamp_ms,
        frame.timestamp_ms,
        frame.image.shape,
    )
    # TODO: Integrate with YOLO.


def process_segment(
    job: SegmentJob,
    storage_root: Path,
    session_factory: sessionmaker[Session],
    *,
    target_fps: float = 2,
) -> int:
    """Decode and select frames for a claimed job, then persist its result."""
    status = SegmentStatus.FAILED
    frame_count = 0
    try:
        with closing(decode_segment(job, storage_root)) as frames:
            for frame in select_frames(frames, target_fps=target_fps):
                frame_count += 1
                if frame_count == 1:
                    logger.debug(
                        "Decoding | segment=%s | shape=%s | timestamp_ms=%s",
                        job.segment_id,
                        frame.image.shape,
                        frame.timestamp_ms,
                    )
                # TODO: YOLO processing goes here

        if frame_count == 0:
            raise ValueError(f"Segment {job.segment_id} contains no selected frames")
        status = SegmentStatus.PROCESSED
    finally:
        with session_factory.begin() as db:
            segment = db.get(VideoSegment, job.segment_id)
            if segment is None:
                raise RuntimeError(f"Segment {job.segment_id} no longer exists")
            update_video_segment_status(db, segment, status=status)

    return frame_count
