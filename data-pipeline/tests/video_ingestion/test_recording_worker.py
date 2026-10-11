import pytest

from video_ingestion.recording_worker import _build_input_args, _prepare_session
from video_ingestion.sources import SourceType, VideoSource


def _make_source(source_type=SourceType.FILE, location="test.mp4", *, enabled=True):
    return VideoSource(
        source_id="camera_01",
        name="Test camera",
        source_type=source_type,
        location=location,
        enabled=enabled,
    )


def test_prepare_session_creates_directory(tmp_path):
    source = _make_source()
    storage_root = tmp_path / "recordings"

    session_id, session_dir = _prepare_session(source, storage_root)

    assert session_dir == storage_root / source.source_id / session_id
    assert session_dir.is_dir()


def test_prepare_session_creates_different_sessions(tmp_path):
    source = _make_source()

    id1, dir1 = _prepare_session(source, tmp_path)
    id2, dir2 = _prepare_session(source, tmp_path)

    assert id1 != id2
    assert dir1.is_dir()
    assert dir2.is_dir()


def test_prepare_session_rejects_disabled_source(tmp_path):
    source = _make_source(enabled=False)
    storage_root = tmp_path / "recordings"

    with pytest.raises(ValueError, match="Video source is disabled"):
        _prepare_session(source, storage_root)

    assert not storage_root.exists()


def test_file_input_args():
    source = _make_source(
        SourceType.FILE,
        "videos/test clip.mp4",
    )

    result = _build_input_args(source)

    assert result == ["-re", "-i", "videos/test clip.mp4"]


def test_rtsp_input_args():
    source = _make_source(
        SourceType.RTSP,
        "rtsp://192.168.1.50:554/live/main",
    )

    result = _build_input_args(source)

    assert result == [
        "-rtsp_transport",
        "tcp",
        "-i",
        "rtsp://192.168.1.50:554/live/main",
    ]


def test_usb_input_is_not_supported_yet():
    source = _make_source(SourceType.USB, "Integrated Camera")

    with pytest.raises(ValueError, match="Unsupported source type"):
        _build_input_args(source)
