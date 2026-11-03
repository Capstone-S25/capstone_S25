from pathlib import Path

import pytest

from data_pipeline.video_ingestion.recording.ffmpeg import (
    build_segment_recording_command,
)


def test_build_command_defaults(make_source, tmp_path):
    source = make_source()
    command = build_segment_recording_command(source, tmp_path)

    assert command[0] == "ffmpeg"
    input_index = command.index("-i")
    assert command[input_index + 1] == source.location
    assert command.index("-re") < input_index
    for option in (
        "-map",
        "-c:v",
        "-an",
        "-f",
        "-segment_format",
        "-segment_time",
        "-reset_timestamps",
        "-segment_list",
        "-segment_list_type",
    ):
        assert command.index(option) > input_index + 1, option

    # Recording behavior
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-c:v") + 1] == "copy"
    assert "-an" in command
    assert command[command.index("-f") + 1] == "segment"
    assert command[command.index("-segment_format") + 1] == "mp4"

    # Default duration and timestamps
    assert command[command.index("-segment_time") + 1] == "5"
    assert command[command.index("-reset_timestamps") + 1] == "1"

    # Manifest and output filename
    assert command[command.index("-segment_list") + 1] == str(
        tmp_path / "completed.csv"
    )
    assert command[command.index("-segment_list_type") + 1] == "csv"
    assert command[-1] == str(tmp_path / "%06d.mp4.part")


def test_build_command_custom_duration(make_source, tmp_path):
    source = make_source()
    command = build_segment_recording_command(source, tmp_path, segment_seconds=10)

    assert command[command.index("-segment_time") + 1] == "10"


@pytest.mark.parametrize("duration", [0, -1])
def test_build_command_rejects_nonpositive_duration(make_source, tmp_path, duration):
    source = make_source()
    with pytest.raises(ValueError):
        build_segment_recording_command(source, tmp_path, segment_seconds=duration)


def test_build_command_resolves_relative_directory(make_source, tmp_path, monkeypatch):
    source = make_source()
    monkeypatch.chdir(tmp_path)

    command = build_segment_recording_command(source, Path("recordings"))

    expected_directory = tmp_path / "recordings"

    assert command[-1] == str(expected_directory / "%06d.mp4.part")
    assert command[command.index("-segment_list") + 1] == str(
        expected_directory / "completed.csv"
    )
