from uuid import UUID

import numpy as np
import pytest

from video_ingestion.decoding.decoder import DecodedFrame


@pytest.fixture
def make_frames():
    """Build decoded frames with explicit millisecond offsets from a UTC anchor."""

    def factory(offsets):
        return [
            DecodedFrame(
                segment_id=1,
                source_id="camera_01",
                session_id=UUID(int=1),
                frame_index=index,
                timestamp_ms=1_700_000_000_000 + offset,
                image=np.zeros((4, 6, 3), dtype=np.uint8),
            )
            for index, offset in enumerate(offsets)
        ]

    return factory
