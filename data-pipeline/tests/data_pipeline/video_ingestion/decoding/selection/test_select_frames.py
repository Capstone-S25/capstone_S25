import pytest

from data_pipeline.video_ingestion.decoding.selection import select_frames


def test_selects_two_fps_from_25_fps_video(make_frames):
    frames = make_frames(range(0, 2000, 40))

    selected = list(select_frames(frames, target_fps=2))

    # 25 FPS provides a frame every 40 ms, so the first >=500 ms gap is 520 ms.
    assert [frame.frame_index for frame in selected] == [0, 13, 26, 39]


def test_preserves_original_frames_and_nonzero_initial_offset(make_frames):
    frames = make_frames([2000, 2400, 2500, 3000])

    selected = list(select_frames(frames, target_fps=2))

    assert [frame.frame_index for frame in selected] == [0, 2, 3]
    for result, original in zip(
        selected, (frames[0], frames[2], frames[3]), strict=True
    ):
        assert result is original
        assert result.image is original.image


def test_handles_uneven_intervals_and_gaps_without_catch_up(make_frames):
    frames = make_frames([0, 100, 499, 510, 900, 1010, 5000, 5010])

    selected = list(select_frames(frames, target_fps=2))

    assert [frame.frame_index for frame in selected] == [0, 3, 5, 6]


def test_low_frame_rate_does_not_duplicate_frames(make_frames):
    frames = make_frames([0, 1000, 2000])

    assert list(select_frames(frames, target_fps=2)) == frames


def test_empty_input_yields_nothing():
    assert list(select_frames([], target_fps=2)) == []


def test_duplicate_timestamps_are_skipped(make_frames):
    selected = list(select_frames(make_frames([0, 0, 500, 500]), target_fps=2))

    assert [frame.frame_index for frame in selected] == [0, 2]


def test_supports_fractional_fps(make_frames):
    selected = list(select_frames(make_frames([0, 1000, 1999, 2000]), target_fps=0.5))

    assert [frame.frame_index for frame in selected] == [0, 3]


@pytest.mark.parametrize("target_fps", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_target_fps(target_fps):
    with pytest.raises(ValueError, match="target_fps"):
        list(select_frames([], target_fps=target_fps))


def test_rejects_decreasing_timestamp_even_on_skipped_frame(make_frames):
    with pytest.raises(ValueError, match="nondecreasing"):
        list(select_frames(make_frames([0, 100, 90]), target_fps=2))


def test_selection_does_not_read_input_ahead(make_frames):
    frames = make_frames([0, 100, 500, 1000])
    consumed = []

    def source():
        for frame in frames:
            consumed.append(frame.frame_index)
            yield frame

    selected = select_frames(source(), target_fps=2)
    assert consumed == []
    assert next(selected) is frames[0]
    assert consumed == [0]
    assert next(selected) is frames[2]
    assert consumed == [0, 1, 2]
    selected.close()
