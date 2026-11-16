import pytest

from data_pipeline.video_ingestion.streaming.selection import FrameSelector


@pytest.mark.parametrize(
    ("timestamps", "target_fps", "expected_indices"),
    [
        (range(0, 2000, 40), 2, [0, 13, 26, 39]),
        ([2000, 2400, 2500, 3000], 2, [0, 2, 3]),
        ([0, 100, 499, 510, 900, 1010, 5000, 5010], 2, [0, 3, 5, 6]),
        ([0, 1000, 2000], 2, [0, 1, 2]),
        ([0, 0, 500, 500], 2, [0, 2]),
        ([0, 1000, 1999, 2000], 0.5, [0, 3]),
    ],
    ids=["25fps", "nonzero_offset", "uneven", "low_rate", "duplicates", "fractional"],
)
def test_selects_by_media_timestamps(timestamps, target_fps, expected_indices):
    selector = FrameSelector(target_fps=target_fps)

    selected_indices = [
        index
        for index, timestamp in enumerate(timestamps)
        if selector.should_select(timestamp)
    ]

    assert selected_indices == expected_indices


@pytest.mark.parametrize("target_fps", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_target_fps(target_fps):
    with pytest.raises(ValueError, match="target_fps"):
        FrameSelector(target_fps=target_fps)


def test_rejects_decreasing_timestamp_even_on_skipped_frame():
    selector = FrameSelector(target_fps=2)
    assert selector.should_select(0)
    assert not selector.should_select(100)

    with pytest.raises(ValueError, match="nondecreasing"):
        selector.should_select(90)
