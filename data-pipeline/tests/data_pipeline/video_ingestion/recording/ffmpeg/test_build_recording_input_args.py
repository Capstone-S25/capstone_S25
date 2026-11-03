import pytest

from data_pipeline.video_ingestion.recording.ffmpeg import _build_recording_input_args
from data_pipeline.video_ingestion.sources import SourceType


def test_file_input_args(make_source):
    source = make_source(
        SourceType.FILE,
        "videos/test clip.mp4",
    )

    result = _build_recording_input_args(source)

    input_index = result.index("-i")
    assert result[input_index + 1] == source.location
    assert result.index("-re") < input_index


def test_rtsp_input_args(make_source):
    source = make_source(
        SourceType.RTSP,
        "rtsp://192.168.1.50:554/live/main",
    )

    result = _build_recording_input_args(source)

    input_index = result.index("-i")
    transport_index = result.index("-rtsp_transport")
    assert result[input_index + 1] == source.location
    assert result[transport_index + 1] == "tcp"
    assert transport_index + 1 < input_index


def test_usb_input_is_not_supported_yet(make_source):
    source = make_source(SourceType.USB, "Integrated Camera")

    with pytest.raises(ValueError):
        _build_recording_input_args(source)
