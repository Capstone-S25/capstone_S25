import pytest

from data_pipeline.video_ingestion.sources import SourceType, VideoSource


@pytest.fixture
def make_source():
    """Build a source with defaults that tests can override."""

    def factory(
        source_type=SourceType.FILE, location="videos/test clip.mp4", *, enabled=True
    ):
        return VideoSource(
            source_id="camera_01",
            name="Test camera",
            source_type=source_type,
            location=location,
            enabled=enabled,
        )

    return factory
