import pytest

from video_ingestion import recording_worker


@pytest.fixture
def entry():
    return recording_worker.SegmentEntry(
        filename="000000.mp4.part", start_seconds=12.0, end_seconds=18.5
    )


def test_finalize_segment_renames_file_and_returns_metadata(tmp_path, entry):
    part_path = tmp_path / entry.filename
    final_path = tmp_path / "000000.mp4"
    part_path.write_bytes(b"temporary segment")

    segment = recording_worker._finalize_segment(
        entry, tmp_path, session_started_at_utc=1_700_000_000_000
    )

    assert not part_path.exists()
    assert final_path.is_file()
    assert final_path.read_bytes() == b"temporary segment"
    assert segment == recording_worker.FinalizedSegment(
        path=final_path,
        start_at_ms=1_700_000_012_000,
        end_at_ms=1_700_000_018_500,
    )


def test_finalize_segment_preserves_both_files_when_destination_exists(tmp_path, entry):
    part_path = tmp_path / entry.filename
    final_path = tmp_path / "000000.mp4"
    part_path.write_bytes(b"temporary segment")
    final_path.write_bytes(b"existing finalized segment")

    with pytest.raises(FileExistsError, match="Final segment already exists"):
        recording_worker._finalize_segment(
            entry, tmp_path, session_started_at_utc=1_700_000_000_000
        )

    assert part_path.read_bytes() == b"temporary segment"
    assert final_path.read_bytes() == b"existing finalized segment"


def test_finalize_segment_rejects_non_temporary_file(tmp_path):
    path = tmp_path / "000000.mp4"
    path.write_bytes(b"existing segment")
    entry = recording_worker.SegmentEntry(path.name, 0.0, 6.0)

    with pytest.raises(ValueError, match="Expected a temporary segment"):
        recording_worker._finalize_segment(
            entry, tmp_path, session_started_at_utc=1_700_000_000_000
        )

    assert path.read_bytes() == b"existing segment"


@pytest.mark.parametrize(
    "start,end,expected_start,expected_end",
    [(0.0, 6.0, 0, 6000), (8.366667, 12.533333, 8367, 12533)],
)
def test_finalize_segment_converts_fractional_offsets(
    tmp_path, start, end, expected_start, expected_end
):
    entry = recording_worker.SegmentEntry("000000.mp4.part", start, end)
    (tmp_path / entry.filename).write_bytes(b"segment")
    segment = recording_worker._finalize_segment(
        entry, tmp_path, session_started_at_utc=1_700_000_000_000
    )
    assert segment.start_at_ms == 1_700_000_000_000 + expected_start
    assert segment.end_at_ms == 1_700_000_000_000 + expected_end
    assert isinstance(segment.start_at_ms, int)
    assert isinstance(segment.end_at_ms, int)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (float("nan"), 1),
        (0, float("nan")),
        (float("inf"), float("inf")),
        (0, float("inf")),
        (-1, 1),
        (2, 1),
        (0, 1e308),
    ],
    ids=[
        "nan-start",
        "nan-end",
        "infinite-start",
        "infinite-end",
        "negative-start",
        "reversed-interval",
        "millisecond-overflow",
    ],
)
def test_invalid_timestamps_leave_temporary_file_untouched(tmp_path, start, end):
    entry = recording_worker.SegmentEntry("000000.mp4.part", start, end)
    part_path = tmp_path / entry.filename
    part_path.write_bytes(b"original segment")

    with pytest.raises(ValueError):
        recording_worker._finalize_segment(
            entry, tmp_path, session_started_at_utc=1_700_000_000_000
        )

    assert part_path.read_bytes() == b"original segment"
    assert not part_path.with_suffix("").exists()
