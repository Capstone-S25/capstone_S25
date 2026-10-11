from unittest.mock import Mock

import pytest

from video_ingestion import recording_worker


@pytest.fixture
def probe(monkeypatch):
    probe = Mock(return_value=6.0)
    monkeypatch.setattr(recording_worker, "_probe_segment_duration", probe)
    return probe


@pytest.fixture
def entry():
    return recording_worker.SegmentEntry(
        filename="000000.mp4.part", start_seconds=12.0, end_seconds=18.5
    )


def test_finalize_segment_renames_file_and_returns_metadata(tmp_path, entry, probe):
    part_path = tmp_path / entry.filename
    final_path = tmp_path / "000000.mp4"
    part_path.write_bytes(b"temporary segment")

    segment = recording_worker._finalize_segment(entry, tmp_path)

    assert not part_path.exists()
    assert final_path.is_file()
    assert final_path.read_bytes() == b"temporary segment"
    assert segment == recording_worker.FinalizedSegment(
        path=final_path,
        start_seconds=12.0,
        end_seconds=18.5,
        duration_seconds=6.0,
    )
    probe.assert_called_once_with(part_path)


def test_finalize_segment_preserves_temporary_file_when_probe_fails(
    tmp_path, entry, probe
):
    part_path = tmp_path / entry.filename
    final_path = tmp_path / "000000.mp4"
    part_path.write_bytes(b"temporary segment")
    probe.side_effect = ValueError("Invalid segment duration")

    with pytest.raises(ValueError, match="Invalid segment duration"):
        recording_worker._finalize_segment(entry, tmp_path)

    assert part_path.read_bytes() == b"temporary segment"
    assert not final_path.exists()
    probe.assert_called_once_with(part_path)


def test_finalize_segment_preserves_both_files_when_destination_exists(
    tmp_path, entry, probe
):
    part_path = tmp_path / entry.filename
    final_path = tmp_path / "000000.mp4"
    part_path.write_bytes(b"temporary segment")
    final_path.write_bytes(b"existing finalized segment")

    with pytest.raises(FileExistsError, match="Final segment already exists"):
        recording_worker._finalize_segment(entry, tmp_path)

    assert part_path.read_bytes() == b"temporary segment"
    assert final_path.read_bytes() == b"existing finalized segment"
    probe.assert_not_called()
