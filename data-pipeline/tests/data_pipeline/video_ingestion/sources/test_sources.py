import pytest

from data_pipeline.video_ingestion.sources import SourceType


def test_source_type_rejects_invalid_value():
    with pytest.raises(ValueError):
        SourceType("rtspp")
