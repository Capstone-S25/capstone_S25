import json
from unittest.mock import Mock

import pytest

from video_ingestion import recording_worker


@pytest.fixture
def ffprobe_run(monkeypatch):
    """Mock FFprobe so tests need neither the executable nor a video file."""
    run = Mock()
    monkeypatch.setattr(recording_worker.subprocess, "run", run)
    return run


def test_probe_segment_duration_returns_positive_duration(ffprobe_run, tmp_path):
    segment_path = tmp_path / "segment.mp4"
    ffprobe_run.return_value.stdout = json.dumps(
        {"streams": [{"index": 0}], "format": {"duration": "6.0"}}
    )

    duration = recording_worker._probe_segment_duration(segment_path)

    assert duration == 6.0
    assert isinstance(duration, float)


def test_probe_segment_duration_rejects_missing_video_stream(ffprobe_run, tmp_path):
    ffprobe_run.return_value.stdout = json.dumps(
        {"streams": [], "format": {"duration": "6.0"}}
    )

    with pytest.raises(ValueError, match="No video stream found"):
        recording_worker._probe_segment_duration(tmp_path / "segment.mp4")


@pytest.mark.parametrize(
    "format_metadata",
    [
        {"duration": "0"},
        {"duration": "-1.0"},
        {"duration": "inf"},
        {"duration": "-inf"},
        {},
        {"duration": "nan"},
    ],
    ids=["zero", "negative", "inf", "negative inf", "missing", "nan"],
)
def test_probe_segment_duration_rejects_invalid_duration(
    ffprobe_run, tmp_path, format_metadata
):
    ffprobe_run.return_value.stdout = json.dumps(
        {"streams": [{"index": 0}], "format": format_metadata}
    )

    with pytest.raises(ValueError, match="Invalid segment duration"):
        recording_worker._probe_segment_duration(tmp_path / "segment.mp4")
