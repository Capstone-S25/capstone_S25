import pytest

from video_ingestion.recording_worker import _prepare_session


def test_prepare_session_creates_directory(make_source, tmp_path):
    source = make_source()
    storage_root = tmp_path / "recordings"

    session_id, session_dir = _prepare_session(source, storage_root)

    assert session_dir == storage_root / source.source_id / session_id
    assert session_dir.is_dir()


def test_prepare_session_creates_different_sessions(make_source, tmp_path):
    source = make_source()

    id1, dir1 = _prepare_session(source, tmp_path)
    id2, dir2 = _prepare_session(source, tmp_path)

    assert id1 != id2
    assert dir1.is_dir()
    assert dir2.is_dir()


def test_prepare_session_rejects_disabled_source(make_source, tmp_path):
    source = make_source(enabled=False)
    storage_root = tmp_path / "recordings"

    with pytest.raises(ValueError):
        _prepare_session(source, storage_root)

    assert not storage_root.exists()
