import logging

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
